# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAM3 Model Wrapper with Custom AutomaticMaskGenerator

SAM3 doesn't have a built-in AutomaticMaskGenerator like SAM2,
so we implement one by sampling a grid of points and calling predict_inst.
"""

import logging
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf


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

        # Debug counters
        _error_count = 0
        _first_error = None
        _score_filtered = 0
        _area_filtered = 0
        _sample_scores = []

        # Process each point
        for point in points_pixel:
            try:
                # Call predict_inst with single point
                # Shape must be (N, 2) for coords and (N,) for labels - SAM adds batch dim
                masks_np, scores_np, low_res_masks = self.model.predict_inst(
                    state,
                    point_coords=np.array([point]),  # Shape: (1, 2)
                    point_labels=np.array([1]),      # Shape: (1,)
                    multimask_output=True,
                    normalize_coords=True,
                )

                # Get best mask (or all 3 if multimask)
                for i in range(len(masks_np)):
                    mask = masks_np[i]
                    score = scores_np[i]
                    area = mask.sum()

                    # Collect sample scores for debugging
                    if len(_sample_scores) < 5:
                        _sample_scores.append((score, area))

                    # Filter by score
                    if score < self.pred_iou_thresh:
                        _score_filtered += 1
                        continue

                    # Filter by area
                    if area < self.min_mask_region_area:
                        _area_filtered += 1
                        continue

                    # Compute stability score from low-res logit masks
                    if low_res_masks is not None and i < len(low_res_masks):
                        stab = calculate_stability_score(
                            low_res_masks[i:i+1],
                            mask_threshold=0.0,
                            stability_score_offset=self.stability_score_offset,
                        )
                        stability = float(stab[0])
                    else:
                        stability = float(score)

                    if stability < self.stability_score_thresh:
                        continue

                    all_masks.append(mask)
                    all_scores.append(score)
                    all_stability.append(stability)

            except Exception as e:
                _error_count += 1
                if _first_error is None:
                    _first_error = str(e)
                continue

        # Debug output
        if len(all_masks) == 0:
            print(f"    SAM3 Debug: {len(points_pixel)} points processed")
            if _error_count > 0:
                print(f"    SAM3 Debug: {_error_count} errors, first: {_first_error}")
            if _sample_scores:
                print(f"    SAM3 Debug: Sample scores (score, area): {_sample_scores}")
            print(f"    SAM3 Debug: Filtered by score (<{self.pred_iou_thresh}): {_score_filtered}")
            print(f"    SAM3 Debug: Filtered by area (<{self.min_mask_region_area}): {_area_filtered}")
            return []

        all_masks = np.array(all_masks)
        all_scores = np.array(all_scores)

        # Apply NMS
        keep_indices = nms_masks(all_masks, all_scores, self.box_nms_thresh)

        all_stability_arr = np.array(all_stability)

        # Build output annotations
        annotations = []
        for idx in keep_indices:
            mask = all_masks[idx]
            score = all_scores[idx]
            stability = all_stability_arr[idx]
            box = mask_to_box(mask)

            annotations.append({
                'segmentation': mask.astype(bool),
                'area': int(mask.sum()),
                'bbox': [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                'predicted_iou': float(score),
                'stability_score': float(stability),
            })

        return annotations


class Sam3Model(nn.Module):
    """
    SAM3 Model wrapper compatible with SAM2Model interface.

    This allows GenerateMasks node to use either SAM2 or SAM3 interchangeably.
    """

    def __init__(self, config: OmegaConf, device='cuda', dtype=None):
        """
        Args:
            config: OmegaConf with sam3.checkpoint path
            device: torch device
            dtype: torch dtype for precision (bf16/fp16/fp32)
        """
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

        self._load_model(dtype=dtype)
        self._setup_engine()

    def _load_model(self, dtype=None):
        """Load SAM3 image model using clean vendored sam3 package."""
        from ...sam3 import build_sam3_image_model
        from ...sam3.utils import Sam3Processor
        from ...sam3.attention import set_sam3_dtype, set_sam3_backend

        checkpoint_path = self.config.sam3.checkpoint

        print(f"[SAM3] Loading image model from: {checkpoint_path}")
        print(f"[SAM3] Building on {self.device} (ModelPatcher moves to GPU before inference), dtype: {dtype}")

        # Configure attention backend for ComfyUI
        set_sam3_backend("auto")

        # Apply precision if specified
        if dtype is not None and dtype in (torch.bfloat16, torch.float16):
            set_sam3_dtype(dtype)

        # Build image model directly (NOT video predictor — saves VRAM)
        self._detector = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            device=self.device,
            eval_mode=True,
            enable_inst_interactivity=True,
            load_from_HF=False,  # Checkpoint already downloaded by SamModelLoader
        )

        # Create processor with proper device
        self._processor = Sam3Processor(
            model=self._detector,
            resolution=1008,
            device=str(self.device),
            confidence_threshold=0.2,
        )

        print(f"[SAM3] Image model loaded successfully")

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

    def process_images_batch(
        self,
        images: list,
        point_grids: list,
        encode_batch_size: int = 6,
    ) -> list:
        """
        Batch-encode images then run per-image AMG decode.

        Uses processor.set_image_batch() for efficient backbone encoding, then
        extracts per-image features and runs point prediction per view.

        Args:
            images: list of np.ndarray (H,W,3) uint8
            point_grids: list of np.ndarray (N,2) normalized [0,1] per image
            encode_batch_size: max images per encoding batch (VRAM limit)

        Returns:
            list of np.ndarray (N_masks, H, W) bool per image
        """
        import time

        engine = self.engine
        processor = self._processor
        detector = self._detector
        predictor = detector.inst_interactive_predictor
        backbone_model = predictor.model

        all_masks_out = []
        t_total = time.time()

        for batch_start in range(0, len(images), encode_batch_size):
            batch_end = min(batch_start + encode_batch_size, len(images))
            sub_images = images[batch_start:batch_end]
            sub_grids = point_grids[batch_start:batch_end]
            batch_label = f"[{batch_start+1}-{batch_end}/{len(images)}]"

            # Convert numpy to PIL for set_image_batch
            pil_images = [Image.fromarray(img) for img in sub_images]

            # Batch encode backbone
            t_enc = time.time()
            state = processor.set_image_batch(pil_images)
            t_enc = time.time() - t_enc
            print(f"      {batch_label} Encoded {len(sub_images)} images in {t_enc:.2f}s")

            # Pre-compute backbone features once for the whole sub-batch
            backbone_out = state["backbone_out"]["sam2_backbone_out"]
            (_, vision_feats, _, feat_sizes) = backbone_model._prepare_backbone_features(backbone_out)
            vision_feats[-1] = vision_feats[-1] + backbone_model.no_mem_embed

            bb_feat_sizes = predictor._bb_feat_sizes

            # Per-image decode
            for local_idx in range(len(sub_images)):
                view_idx = batch_start + local_idx
                h, w = sub_images[local_idx].shape[:2]

                pg = sub_grids[local_idx]
                if len(pg) == 0:
                    all_masks_out.append(np.zeros((1, h, w), dtype=bool))
                    print(f"      View {view_idx+1}/{len(images)}: skipped (no valid points)")
                    continue

                t_view = time.time()

                # Extract per-image features from batch (dim 1 is batch)
                vision_feats_single = [feat[:, local_idx:local_idx+1, :] for feat in vision_feats]

                feats = [
                    feat.permute(1, 2, 0).view(1, -1, *fs)
                    for feat, fs in zip(vision_feats_single[::-1], bb_feat_sizes[::-1])
                ][::-1]

                # Set features on predictor for this image
                predictor._features = {
                    "image_embed": feats[-1],
                    "high_res_feats": feats[:-1],
                }
                predictor._is_image_set = True
                predictor._orig_hw = [
                    (state["original_heights"][local_idx], state["original_widths"][local_idx])
                ]

                # Scale normalized points to pixel coords
                points_pixel = pg.copy()
                points_pixel[:, 0] *= w
                points_pixel[:, 1] *= h

                # Batched point prediction — call _predict() directly with
                # (N, 1, 2) shape so each point is a separate prompt.
                # This replaces 1024 individual predict() calls with ~16 batched calls.
                points_per_batch = 64  # conservative for 1024×1024 images
                orig_hw = predictor._orig_hw[0]

                points_tensor = torch.as_tensor(
                    points_pixel, dtype=torch.float32, device=predictor.device
                )
                transformed_pts = predictor._transforms.transform_coords(
                    points_tensor, normalize=True, orig_hw=orig_hw
                )

                # Cast features to fp32 for batched decode (repeat_image=True has bf16 issues)
                if self.dtype and self.dtype != torch.float32:
                    predictor._features = {
                        "image_embed": predictor._features["image_embed"].float(),
                        "high_res_feats": [f.float() for f in predictor._features["high_res_feats"]],
                    }

                view_masks = []
                view_scores = []

                for bp_start in range(0, len(transformed_pts), points_per_batch):
                    bp_end = min(bp_start + points_per_batch, len(transformed_pts))
                    batch_pts = transformed_pts[bp_start:bp_end]
                    batch_labels = torch.ones(
                        len(batch_pts), dtype=torch.int, device=predictor.device
                    )

                    # (N, 1, 2) = N separate single-point prompts
                    masks_t, iou_preds_t, low_res_t = predictor._predict(
                        batch_pts[:, None, :],
                        batch_labels[:, None],
                        multimask_output=True,
                        return_logits=True,
                    )
                    # masks_t: (N, 3, H, W) logits, iou_preds_t: (N, 3)

                    # Flatten to (N*3,) for filtering
                    n_pts = masks_t.shape[0]
                    n_multi = masks_t.shape[1]
                    iou_flat = iou_preds_t.reshape(-1)  # (N*3,)
                    low_res_flat = low_res_t.reshape(-1, *low_res_t.shape[2:])  # (N*3, 64, 64)
                    masks_flat = masks_t.reshape(-1, *masks_t.shape[2:])  # (N*3, H, W)

                    # Filter by IoU
                    keep_iou = iou_flat > engine.pred_iou_thresh
                    if not keep_iou.any():
                        del masks_t, iou_preds_t, low_res_t
                        continue

                    iou_kept = iou_flat[keep_iou]
                    low_res_kept = low_res_flat[keep_iou]
                    masks_kept = masks_flat[keep_iou]

                    # Stability score from low-res logits
                    high = (low_res_kept > 0.0).sum(dim=(-2, -1)).float()
                    low = (low_res_kept > -engine.stability_score_offset).sum(dim=(-2, -1)).float()
                    stability = torch.where(low > 0, high / low, torch.zeros_like(high))

                    keep_stab = stability >= engine.stability_score_thresh
                    if not keep_stab.any():
                        del masks_t, iou_preds_t, low_res_t
                        continue

                    iou_kept = iou_kept[keep_stab]
                    masks_kept = masks_kept[keep_stab]

                    # Threshold logits to boolean
                    masks_bool = (masks_kept > 0.0).cpu().numpy()
                    iou_np = iou_kept.cpu().numpy()

                    # Filter by area
                    for i in range(len(masks_bool)):
                        area = masks_bool[i].sum()
                        if area >= engine.min_mask_region_area:
                            view_masks.append(masks_bool[i])
                            view_scores.append(float(iou_np[i]))

                    del masks_t, iou_preds_t, low_res_t

                # NMS
                n_masks = 0
                if view_masks:
                    masks_arr = np.array(view_masks)
                    scores_arr = np.array(view_scores)
                    keep = nms_masks(masks_arr, scores_arr, engine.box_nms_thresh)
                    kept = masks_arr[keep]
                    kept = sorted(kept, key=lambda m: m.sum(), reverse=True)
                    all_masks_out.append(np.stack(kept))
                    n_masks = len(kept)
                else:
                    all_masks_out.append(np.zeros((1, h, w), dtype=bool))

                t_view = time.time() - t_view
                print(f"      View {view_idx+1}/{len(images)}: {n_masks} masks in {t_view:.2f}s")

            # Cleanup predictor state
            predictor._features = None
            predictor._is_image_set = False

        t_total = time.time() - t_total
        print(f"      Total batch: {len(images)} views in {t_total:.2f}s ({t_total/len(images):.2f}s/view)")

        return all_masks_out

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
