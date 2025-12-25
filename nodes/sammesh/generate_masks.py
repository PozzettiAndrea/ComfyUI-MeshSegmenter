# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Generate Masks Node - Loads SAM2 and runs it on rendered images.
"""

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from .types import VIEW_MASKS
from ...samesh.models.mask_utils import (
    combine_bmasks,
    remove_artifacts,
    colormap_mask,
    visualize_points
)


def tensor_to_pil_list(tensor: torch.Tensor) -> list:
    """Convert ComfyUI image tensor (B, H, W, C) to list of PIL Images."""
    images = []
    arr = tensor.numpy()
    for i in range(arr.shape[0]):
        img_arr = (arr[i] * 255).astype(np.uint8)
        images.append(Image.fromarray(img_arr))
    return images


def numpy_to_tensor(arrays: list, normalize: bool = True) -> torch.Tensor:
    """Convert list of numpy arrays to ComfyUI image tensor."""
    tensors = []
    for arr in arrays:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if normalize and arr.max() > 1:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        arr = np.clip(arr, 0, 1)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        tensors.append(arr)
    if not tensors:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    return torch.from_numpy(np.stack(tensors, axis=0))


# Cache for loaded SAM model to avoid reloading
_sam_model_cache = {}


class GenerateMasks:
    """
    Loads SAM2 and runs it on rendered images to generate masks.

    Input order matches MultiViewRenderer output order for clean wiring:
    - sam_checkpoint_path, sam_model_config_path: SAM2 model files
    - normals: RGB normal maps (optional)
    - matte: RGB shaded renders (optional)
    - sdf: RGB SDF renders (optional)
    - face_mask: Single channel MASK for point sampling (optional)

    At least one of normals/matte/sdf is required.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_checkpoint_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to SAM2 model checkpoint (.pt file)"
                }),
                "sam_model_config_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to SAM2 model config (.yaml file)"
                }),
            },
            "optional": {
                # Image inputs - order matches MultiViewRenderer outputs
                "normals": ("IMAGE", {
                    "tooltip": "RGB normal maps from MultiViewRenderer"
                }),
                "matte": ("IMAGE", {
                    "tooltip": "RGB shaded renders from MultiViewRenderer"
                }),
                "sdf": ("IMAGE", {
                    "tooltip": "RGB SDF renders from MultiViewRenderer"
                }),
                "face_mask": ("MASK", {
                    "tooltip": "Foreground mask from MultiViewRenderer (1=mesh, 0=background)"
                }),
                # SAM parameters
                "points_per_side": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 64,
                    "step": 8,
                    "tooltip": "Grid density for SAM point sampling."
                }),
                "pred_iou_thresh": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Minimum predicted IoU score to keep a mask."
                }),
                "stability_score_thresh": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Minimum stability score to keep a mask."
                }),
                "min_area": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Minimum mask area to keep."
                }),
                "show_points": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show point visualization (green=valid, red=invalid)."
                }),
            }
        }

    RETURN_TYPES = (VIEW_MASKS, "IMAGE", "IMAGE")
    RETURN_NAMES = ("masks", "colored_masks", "point_viz")
    FUNCTION = "generate_masks"
    CATEGORY = "meshsegmenter/sammesh"

    def _load_sam_model(
        self,
        checkpoint_path: str,
        config_path: str,
        points_per_side: int,
        pred_iou_thresh: float,
        stability_score_thresh: float
    ):
        """Load SAM model, using cache if available."""
        cache_key = (checkpoint_path, config_path)

        if cache_key in _sam_model_cache:
            model = _sam_model_cache[cache_key]
            # Update engine config
            model.engine.points_per_side = points_per_side
            model.engine.pred_iou_thresh = pred_iou_thresh
            model.engine.stability_score_thresh = stability_score_thresh
            print(f"GenerateMasks: Using cached SAM model")
            return model

        # Validate paths
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SAM config not found: {config_path}")

        print(f"GenerateMasks: Loading SAM2 model...")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Config: {config_path}")

        engine_config = {
            "points_per_side": points_per_side,
            "crop_n_layers": 0,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "stability_score_offset": 1.0,
        }

        config = OmegaConf.create({
            "sam": {
                "checkpoint": checkpoint_path,
                "model_config": os.path.basename(config_path),
                "auto": True,
                "ground": False,
                "engine_config": engine_config,
            }
        })

        from ...samesh.models.sam import Sam2Model
        model = Sam2Model(config, device='cuda')

        _sam_model_cache[cache_key] = model
        print(f"  Model loaded and cached!")

        return model

    def generate_masks(
        self,
        sam_checkpoint_path: str,
        sam_model_config_path: str,
        normals: torch.Tensor = None,
        matte: torch.Tensor = None,
        sdf: torch.Tensor = None,
        face_mask: torch.Tensor = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.5,
        stability_score_thresh: float = 0.7,
        min_area: int = 1024,
        show_points: bool = True
    ):
        from ...samesh.models.sam import point_grid_from_mask

        # Collect all provided image inputs
        image_inputs = []
        image_names = []
        if normals is not None:
            image_inputs.append(('normals', normals))
            image_names.append('normals')
        if matte is not None:
            image_inputs.append(('matte', matte))
            image_names.append('matte')
        if sdf is not None:
            # Check if SDF is all black (compute_sdf was False)
            if sdf.sum() > 0:
                image_inputs.append(('sdf', sdf))
                image_names.append('sdf')

        if not image_inputs:
            raise ValueError("At least one image input (normals, matte, or sdf) is required!")

        print(f"GenerateMasks: Processing {len(image_inputs)} image types: {image_names}")
        print(f"  Points per side: {points_per_side}")
        print(f"  Min area: {min_area}")

        # Load SAM model
        model = self._load_sam_model(
            sam_checkpoint_path,
            sam_model_config_path,
            points_per_side,
            pred_iou_thresh,
            stability_score_thresh
        )

        # Get dimensions from first image input
        first_img = image_inputs[0][1]
        n_views = first_img.shape[0]
        h, w = first_img.shape[1], first_img.shape[2]

        # Convert face_mask to numpy, or create full mask if not provided
        if face_mask is not None:
            face_mask_np = face_mask.numpy()
            print(f"  Using provided face_mask for point sampling")
        else:
            # Auto-detect from image brightness (black = background)
            first_arr = first_img.numpy()
            face_mask_np = (first_arr.mean(axis=-1) > 0.01).astype(np.float32)
            print(f"  Auto-detecting foreground from image brightness")

        # Process each image type
        all_bmasks = []
        all_point_status = []

        # Initialize per-view storage
        for _ in range(n_views):
            all_bmasks.append([])
            all_point_status.append({'valid': [], 'invalid': []})

        # For visualization, use the first available image type
        viz_images = tensor_to_pil_list(image_inputs[0][1])

        for img_name, img_tensor in image_inputs:
            print(f"  Processing {img_name}...")
            pil_images = tensor_to_pil_list(img_tensor)

            for view_idx, image in enumerate(tqdm(pil_images, desc=f"    SAM on {img_name}")):
                img_arr = np.array(image)
                h, w = img_arr.shape[:2]

                # Get valid mask for this view
                valid_mask = face_mask_np[view_idx] > 0.5

                # Sample point grid within valid region
                try:
                    point_grid = point_grid_from_mask(valid_mask, points_per_side ** 2)
                    model.engine.point_grids = [point_grid]
                except ValueError:
                    point_grid = np.array([])

                # Run SAM
                try:
                    bmasks = model(image)
                except Exception as e:
                    print(f"      Warning: SAM failed on view {view_idx}: {e}")
                    bmasks = np.zeros((1, h, w), dtype=bool)

                # Track valid/invalid points (only for first image type)
                if img_name == image_names[0] and len(point_grid) > 0 and show_points:
                    for px, py in point_grid:
                        pixel_x = int(px * (w - 1))
                        pixel_y = int(py * (h - 1))
                        point_produced_mask = False
                        for mask in bmasks:
                            if mask[pixel_y, pixel_x] and mask.sum() >= min_area:
                                point_produced_mask = True
                                break
                        if point_produced_mask:
                            all_point_status[view_idx]['valid'].append((pixel_x, pixel_y))
                        else:
                            all_point_status[view_idx]['invalid'].append((pixel_x, pixel_y))

                # Filter by area and add to view's mask list
                filtered = [m for m in bmasks if m.sum() >= min_area]
                if filtered:
                    all_bmasks[view_idx].extend(filtered)

        # Combine masks per view
        combined_bmasks = []
        combined_cmasks = []
        colored_masks_list = []
        point_viz_list = []

        for view_idx in range(n_views):
            h, w = face_mask_np[view_idx].shape

            if all_bmasks[view_idx]:
                view_bmasks = np.array(all_bmasks[view_idx])
                cmask = combine_bmasks(view_bmasks, sort=True)
                cmask = remove_artifacts(cmask, mode='islands', min_area=min_area // 4)
                cmask = remove_artifacts(cmask, mode='holes', min_area=min_area // 4)
            else:
                view_bmasks = np.zeros((1, h, w), dtype=bool)
                cmask = np.zeros((h, w), dtype=int)

            combined_bmasks.append(view_bmasks)
            combined_cmasks.append(cmask)

            # Colored mask visualization
            colored = colormap_mask(cmask, seed=42)
            colored_masks_list.append(np.array(colored))

            # Point visualization
            if show_points:
                viz_arr = np.array(viz_images[view_idx])
                point_viz = visualize_points(
                    viz_arr.copy(),
                    all_point_status[view_idx]['valid'],
                    all_point_status[view_idx]['invalid'],
                    radius=4,
                    valid_color=(0, 255, 0),
                    invalid_color=(255, 0, 0)
                )
                point_viz_list.append(point_viz)
            else:
                point_viz_list.append(np.array(viz_images[view_idx]))

        # Build outputs
        masks_data = {
            'bmasks': combined_bmasks,
            'cmasks': combined_cmasks,
            'point_status': all_point_status,
        }

        colored_masks_tensor = numpy_to_tensor(colored_masks_list, normalize=True)
        point_viz_tensor = numpy_to_tensor(point_viz_list, normalize=True)

        total_masks = sum(len(bm) for bm in combined_bmasks)
        print(f"  Generated {total_masks} masks across {n_views} views")

        return (masks_data, colored_masks_tensor, point_viz_tensor)
