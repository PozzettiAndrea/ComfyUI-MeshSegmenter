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


def parse_channel_config(config: str, available_channels: list) -> list:
    """
    Parse channel config and return list of [ch1, ch2, ch3] triplets.

    Config format (YAML-like):
        - normal_x, normal_y, normal_z
        - matte_r, matte_g, matte_b
        - sdf, sdf, sdf

    Each line starting with '-' defines one RGB image from 3 channel names.
    Raises error if channel not found.
    """
    images = []
    for line in config.strip().split('\n'):
        line = line.strip()
        if not line or not line.startswith('-'):
            continue
        # Remove leading dash and split by comma
        channels = [c.strip() for c in line[1:].strip().split(',')]
        if len(channels) != 3:
            raise ValueError(f"Each line must have exactly 3 channels, got: {channels}")
        # Validate channels exist
        for ch in channels:
            if ch not in available_channels:
                raise ValueError(f"Channel '{ch}' not found. Available: {available_channels}")
        images.append(channels)
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
                # Single multiband input from MultiViewRenderer
                "renders": ("MULTIBAND_IMAGE", {
                    "tooltip": "Combined renders from MultiViewRenderer (normals, matte, sdf, mask channels)"
                }),
                # Channel config
                "channel_config": ("STRING", {
                    "default": "- normal_x, normal_y, normal_z\n- matte_r, matte_g, matte_b\n- sdf, sdf, sdf",
                    "multiline": True,
                    "tooltip": "YAML-like config: each line defines an RGB image from 3 channel names. Use same channel 3x for grayscale (e.g. sdf, sdf, sdf)"
                }),
                "mask_channel": ("STRING", {
                    "default": "mask",
                    "tooltip": "Channel name to use for point sampling mask"
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
        renders: dict = None,
        channel_config: str = "- normal_x, normal_y, normal_z\n- matte_r, matte_g, matte_b\n- sdf, sdf, sdf",
        mask_channel: str = "mask",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.5,
        stability_score_thresh: float = 0.7,
        min_area: int = 1024,
        show_points: bool = True
    ):
        from ...samesh.models.sam import point_grid_from_mask

        if renders is None:
            raise ValueError("renders (MULTIBAND_IMAGE) input is required!")

        # Extract channels from MULTIBAND_IMAGE
        samples = renders['samples']  # (B, C, H, W)
        channel_list = renders.get('channel_names', [])

        print(f"GenerateMasks: Received MULTIBAND_IMAGE with {samples.shape[1]} channels: {channel_list}")

        # Parse channel config to get image specs
        image_specs = parse_channel_config(channel_config, channel_list)
        if not image_specs:
            raise ValueError("No valid image specs in channel_config!")

        print(f"GenerateMasks: Building {len(image_specs)} images from config")

        # Build images from config
        image_inputs = []
        image_names = []
        for ch1, ch2, ch3 in image_specs:
            idx1 = channel_list.index(ch1)
            idx2 = channel_list.index(ch2)
            idx3 = channel_list.index(ch3)

            # Extract and stack channels: (B,3,H,W) -> (B,H,W,3)
            img = torch.stack([
                samples[:, idx1, :, :],
                samples[:, idx2, :, :],
                samples[:, idx3, :, :],
            ], dim=-1)

            image_name = f"{ch1}+{ch2}+{ch3}"
            image_inputs.append((image_name, img))
            image_names.append(image_name)
            print(f"  Image '{image_name}' from channels [{idx1}, {idx2}, {idx3}]")

        # Extract mask channel for point sampling
        if mask_channel not in channel_list:
            raise ValueError(f"Mask channel '{mask_channel}' not found. Available: {channel_list}")
        mask_idx = channel_list.index(mask_channel)
        face_mask_tensor = samples[:, mask_idx, :, :]  # (B,H,W)

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
        if face_mask_tensor is not None:
            face_mask_np = face_mask_tensor.numpy()
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
