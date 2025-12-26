# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Combine BMasks Node - Combines multiple binary mask channels into a single labeled mask.
"""

import numpy as np
import torch

from ...samesh.models.mask_utils import combine_bmasks, remove_artifacts


class CombineBMasks:
    """
    Combines multiple binary mask channels into a single labeled mask.

    Takes a MULTIBAND_IMAGE with multiple binary mask channels (e.g., seg_00, seg_01, seg_02...)
    and combines them into a single labeled mask where each original mask gets a unique label.

    Masks are sorted by area (largest first) so smaller masks can occlude larger ones.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MULTIBAND_IMAGE", {
                    "tooltip": "Multiband image with binary mask channels to combine"
                }),
            },
            "optional": {
                "sort_by_area": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Sort masks by area (largest first, so smaller masks paint on top)"
                }),
                "remove_islands": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove small disconnected regions"
                }),
                "remove_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill small holes in masks"
                }),
                "min_area": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Minimum area for island/hole removal"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Threshold for converting float masks to binary"
                }),
            }
        }

    RETURN_TYPES = ("MULTIBAND_IMAGE",)
    RETURN_NAMES = ("combined_mask",)
    FUNCTION = "combine"
    CATEGORY = "meshsegmenter/sammesh"

    def combine(
        self,
        masks: dict,
        sort_by_area: bool = True,
        remove_islands: bool = True,
        remove_holes: bool = True,
        min_area: int = 256,
        threshold: float = 0.5,
    ):
        samples = masks['samples']  # (B, C, H, W)
        channel_names = masks.get('channel_names', [])

        B, C, H, W = samples.shape

        print(f"CombineBMasks: Combining {C} mask channels across {B} views")
        print(f"  Channel names: {channel_names}")
        print(f"  Sort by area: {sort_by_area}")
        print(f"  Threshold: {threshold}")

        combined_masks = []

        for batch_idx in range(B):
            # Extract all binary masks for this view
            bmasks = []
            for ch_idx in range(C):
                mask = samples[batch_idx, ch_idx].numpy()

                # Convert to binary using threshold
                binary_mask = mask > threshold

                # Skip empty masks
                if binary_mask.sum() > 0:
                    bmasks.append(binary_mask)

            if bmasks:
                bmasks_arr = np.array(bmasks)

                # Combine into single labeled mask
                cmask = combine_bmasks(bmasks_arr, sort=sort_by_area)

                # Remove artifacts if requested
                if remove_islands and min_area > 0:
                    cmask = remove_artifacts(cmask, mode='islands', min_area=min_area)
                if remove_holes and min_area > 0:
                    cmask = remove_artifacts(cmask, mode='holes', min_area=min_area)
            else:
                cmask = np.zeros((H, W), dtype=np.int32)

            combined_masks.append(cmask)

        # Stack into tensor
        combined_np = np.stack(combined_masks, axis=0).astype(np.float32)  # (B, H, W)
        combined_tensor = torch.from_numpy(combined_np).unsqueeze(1)  # (B, 1, H, W)

        # Get unique labels count
        unique_labels = len(np.unique(combined_np)) - 1  # exclude background (0)
        print(f"  Combined into {unique_labels} unique labels")

        # Create output MULTIBAND_IMAGE
        result = {
            'samples': combined_tensor,
            'channel_names': ['combined_mask'],
            'metadata': {
                'source': 'combine_bmasks',
                'n_views': B,
                'n_input_channels': C,
                'n_labels': unique_labels,
                'sorted_by_area': sort_by_area,
            }
        }

        return (result,)
