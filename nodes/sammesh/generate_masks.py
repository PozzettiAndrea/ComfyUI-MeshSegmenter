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

from .types import SAM_MODEL
from ...samesh.models.mask_utils import (
    combine_bmasks,
    remove_artifacts,
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


class GenerateMasks:
    """
    Runs SAM2 on rendered images to generate masks.

    Takes a pre-loaded SAM model from SamModelLoader node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": (SAM_MODEL, {
                    "tooltip": "Loaded SAM2 model from SamModelLoader node"
                }),
            },
            "optional": {
                # Single multiband input from MultiViewRenderer
                "renders": ("MULTIBAND_IMAGE", {
                    "tooltip": "Combined renders from MultiViewRenderer (normals, matte, sdf, mask channels)"
                }),
                # Channel config
                "channel_config": ("STRING", {
                    "default": "- normal_x, normal_y, normal_z\n- matte, matte, matte\n- sdf, sdf, sdf",
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
            }
        }

    RETURN_TYPES = ("MULTIBAND_IMAGE",)
    RETURN_NAMES = ("binary_masks",)
    FUNCTION = "generate_masks"
    CATEGORY = "meshsegmenter/sammesh"

    def generate_masks(
        self,
        sam_model,
        renders: dict = None,
        channel_config: str = "- normal_x, normal_y, normal_z\n- matte, matte, matte\n- sdf, sdf, sdf",
        mask_channel: str = "mask",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.5,
        stability_score_thresh: float = 0.7,
        min_area: int = 1024,
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

        # Update model engine config with current parameters
        sam_model.engine.points_per_side = points_per_side
        sam_model.engine.pred_iou_thresh = pred_iou_thresh
        sam_model.engine.stability_score_thresh = stability_score_thresh
        model = sam_model

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

        # Process each input type separately, combine masks per type
        # all_bmasks_per_type[type_idx][view_idx] = list of binary masks
        n_types = len(image_inputs)
        all_bmasks_per_type = [[[] for _ in range(n_views)] for _ in range(n_types)]

        for type_idx, (img_name, img_tensor) in enumerate(image_inputs):
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

                # Filter by area and add to this type's view mask list
                filtered = [m for m in bmasks if m.sum() >= min_area]
                if filtered:
                    all_bmasks_per_type[type_idx][view_idx].extend(filtered)

        # Combine masks per type into segmentations
        seg_channels = []
        for type_idx in range(n_types):
            type_cmasks = []
            for view_idx in range(n_views):
                h, w = face_mask_np[view_idx].shape
                view_bmasks = all_bmasks_per_type[type_idx][view_idx]

                if view_bmasks:
                    bmasks_arr = np.array(view_bmasks)
                    cmask = combine_bmasks(bmasks_arr, sort=True)

                    # Shift labels, mask background
                    cmask = cmask + 1
                    background = face_mask_np[view_idx] < 0.5
                    cmask[background] = 0

                    cmask = remove_artifacts(cmask, mode='islands', min_area=min_area // 4)
                    cmask = remove_artifacts(cmask, mode='holes', min_area=min_area // 4)
                else:
                    cmask = np.zeros((h, w), dtype=np.int32)

                type_cmasks.append(cmask)

            # Stack views for this type: (n_views, H, W)
            seg_channels.append(torch.from_numpy(np.stack(type_cmasks, axis=0).astype(np.float32)))

        # Stack types: (n_views, n_types, H, W)
        seg_tensor = torch.stack(seg_channels, dim=1)

        # Add foreground mask as last channel
        fg_mask_tensor = face_mask_tensor.unsqueeze(1)  # (n_views, 1, H, W)
        combined_tensor = torch.cat([seg_tensor, fg_mask_tensor], dim=1)  # (n_views, n_types+1, H, W)

        # Create channel names: seg_00, seg_01, ..., foreground_mask
        channel_names = [f"seg_{i:02d}" for i in range(n_types)] + ["foreground_mask"]

        # Create MULTIBAND_IMAGE
        output_multiband = {
            'samples': combined_tensor,
            'channel_names': channel_names,
            'metadata': {
                'source': 'generate_masks',
                'n_views': n_views,
                'n_types': n_types,
                'input_types': image_names,
            }
        }

        total_masks = sum(len(bm) for type_bmasks in all_bmasks_per_type for bm in type_bmasks)
        print(f"  Generated {total_masks} masks across {n_views} views from {n_types} input types")
        print(f"  Output: {n_types} segmentation channels + foreground_mask")

        return (output_multiband,)
