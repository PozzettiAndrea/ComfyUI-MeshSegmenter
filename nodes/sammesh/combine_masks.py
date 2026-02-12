# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Combine View Masks Node - Merges masks from different modes (normals, SDF, matte).
"""

import numpy as np
import torch

from .types import VIEW_MASKS, MESH_RENDER_DATA
from ...samesh.models.mask_utils import combine_bmasks, remove_artifacts, colormap_mask


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


class CombineViewMasks:
    """
    Combines masks from multiple input sources (normals, SDF, matte) into
    a unified set of masks per view.

    This allows using multiple cues for more robust segmentation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "render_data": (MESH_RENDER_DATA,),
            },
            "optional": {
                "masks_normals": (VIEW_MASKS, {
                    "tooltip": "Masks generated from normal map renders."
                }),
                "masks_sdf": (VIEW_MASKS, {
                    "tooltip": "Masks generated from SDF renders."
                }),
                "masks_matte": (VIEW_MASKS, {
                    "tooltip": "Masks generated from matte/shaded renders."
                }),
                "min_area": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Minimum area for artifact removal."
                }),
            }
        }

    RETURN_TYPES = (VIEW_MASKS, "IMAGE")
    RETURN_NAMES = ("combined_masks", "masks_visualization")
    FUNCTION = "combine_masks"
    CATEGORY = "meshsegmenter/sammesh"

    def combine_masks(
        self,
        render_data: dict,
        masks_normals: dict = None,
        masks_sdf: dict = None,
        masks_matte: dict = None,
        min_area: int = 1024
    ):
        print("CombineViewMasks: Combining masks from multiple sources...")

        # Get face ID renders for background masking
        faces_list = render_data['faces']
        n_views = len(faces_list)

        # Collect all mask sources
        sources = []
        source_names = []
        if masks_normals is not None:
            sources.append(masks_normals)
            source_names.append('normals')
        if masks_sdf is not None:
            sources.append(masks_sdf)
            source_names.append('sdf')
        if masks_matte is not None:
            sources.append(masks_matte)
            source_names.append('matte')

        if not sources:
            raise ValueError("At least one mask source must be provided!")

        print(f"  Sources: {source_names}")
        print(f"  Views: {n_views}")

        # Combine binary masks from all sources
        combined_bmasks = []
        combined_cmasks = []
        combined_point_status = []

        for view_idx in range(n_views):
            # Concatenate all binary masks for this view
            view_bmasks = []
            for source in sources:
                if view_idx < len(source['bmasks']):
                    view_bmasks.append(source['bmasks'][view_idx])

            if view_bmasks:
                all_bmasks = np.concatenate(view_bmasks, axis=0)
            else:
                # Create empty mask
                h, w = faces_list[view_idx].shape
                all_bmasks = np.zeros((1, h, w), dtype=bool)

            # Combine into labeled mask
            cmask = combine_bmasks(all_bmasks, sort=True)

            # Shift labels so background is 0
            cmask = cmask + 1

            # Mask out background using face IDs
            faces = faces_list[view_idx]
            cmask[faces == -1] = 0

            # Remove artifacts
            cmask = remove_artifacts(cmask, mode='islands', min_area=min_area)
            cmask = remove_artifacts(cmask, mode='holes', min_area=min_area)

            combined_bmasks.append(all_bmasks)
            combined_cmasks.append(cmask)

            # Merge point status from all sources
            merged_valid = []
            merged_invalid = []
            for source in sources:
                if view_idx < len(source.get('point_status', [])):
                    ps = source['point_status'][view_idx]
                    merged_valid.extend(ps.get('valid', []))
                    merged_invalid.extend(ps.get('invalid', []))
            combined_point_status.append({
                'valid': merged_valid,
                'invalid': merged_invalid
            })

        # Build output VIEW_MASKS
        masks_data = {
            'bmasks': combined_bmasks,
            'cmasks': combined_cmasks,
            'point_status': combined_point_status,
        }

        # Create visualization
        colored_masks_list = []
        for cmask in combined_cmasks:
            colored = colormap_mask(cmask, seed=42)
            colored_masks_list.append(np.array(colored))

        masks_viz = numpy_to_tensor(colored_masks_list, normalize=True)

        total_labels = sum(int(cm.max()) for cm in combined_cmasks)
        print(f"  Combined {len(sources)} sources into {total_labels} total labels")
        print(f"  Output shape: {masks_viz.shape}")

        return (masks_data, masks_viz)
