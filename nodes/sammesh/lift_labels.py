# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Lift 2D to 3D Labels Node - Converts per-view 2D masks to consistent 3D face labels.
"""

import numpy as np

from .types import CAMERA_POSES, FACE_LABELS
from ...samesh.models.lifting import lift_masks_to_3d


class Lift2DTo3DLabels:
    """
    Lifts 2D per-view mask labels to consistent 3D mesh face labels.

    This node:
    1. Computes face-to-label mapping for each view
    2. Builds a match graph connecting labels across views
    3. Uses community detection to find consistent label groups
    4. Outputs a single face-to-label mapping for the entire mesh
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "renders": ("MULTIBAND_IMAGE", {
                    "tooltip": "Renders from MultiViewRenderer (contains face_id, normals)"
                }),
                "segmentations": ("MULTIBAND_IMAGE", {
                    "tooltip": "Segmentations from GenerateMasks (contains seg_00, seg_01, etc.)"
                }),
                "poses": (CAMERA_POSES, {
                    "tooltip": "Camera poses from MultiViewRenderer"
                }),
            },
            "optional": {
                "seg_channel": ("STRING", {
                    "default": "seg_00",
                    "tooltip": "Which segmentation channel to use for lifting (e.g. seg_00, seg_01)"
                }),
                "connections_threshold": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Minimum overlapping faces to form a connection between views."
                }),
                "face2label_threshold": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Minimum pixel count for a face to be assigned a label."
                }),
            }
        }

    RETURN_TYPES = (FACE_LABELS,)
    RETURN_NAMES = ("face_labels",)
    FUNCTION = "lift_labels"
    CATEGORY = "meshsegmenter/sammesh"

    def lift_labels(
        self,
        mesh,
        renders: dict,
        segmentations: dict,
        poses: np.ndarray,
        seg_channel: str = "seg_00",
        connections_threshold: int = 8,
        face2label_threshold: int = 4
    ):
        print("Lift2DTo3DLabels: Lifting 2D masks to 3D face labels...")

        # Extract data from renders MULTIBAND_IMAGE
        render_samples = renders['samples']  # (B, C, H, W)
        render_channels = renders.get('channel_names', [])

        # Get face_id channel
        if 'face_id' not in render_channels:
            raise ValueError(f"'face_id' channel not found in renders. Available: {render_channels}")
        face_id_idx = render_channels.index('face_id')
        face_id_tensor = render_samples[:, face_id_idx, :, :]  # (B, H, W)

        # Get normal channels (x, y, z)
        if not all(c in render_channels for c in ['normal_x', 'normal_y', 'normal_z']):
            raise ValueError(f"Normal channels not found in renders. Available: {render_channels}")
        nx_idx = render_channels.index('normal_x')
        ny_idx = render_channels.index('normal_y')
        nz_idx = render_channels.index('normal_z')

        # Stack normals: (B, H, W, 3)
        # Note: stored as RGB colorized (0-1), need to convert back to raw normals (-1,1)
        # colormap_norms does: rgb = (raw + 1) / 2, so raw = rgb * 2 - 1
        normals_rgb = np.stack([
            render_samples[:, nx_idx, :, :].numpy(),
            render_samples[:, ny_idx, :, :].numpy(),
            render_samples[:, nz_idx, :, :].numpy(),
        ], axis=-1)
        normals_tensor = normals_rgb * 2 - 1  # Convert RGB (0-1) back to raw normals (-1,1)

        # Convert face_id to list of numpy arrays (integers)
        face_id_np = face_id_tensor.numpy()
        faces_list = [face_id_np[i].astype(np.int32) for i in range(face_id_np.shape[0])]

        # Convert normals to list
        norms_list = [normals_tensor[i] for i in range(normals_tensor.shape[0])]

        # Extract segmentation channel from segmentations MULTIBAND_IMAGE
        seg_samples = segmentations['samples']  # (B, C, H, W)
        seg_channels = segmentations.get('channel_names', [])

        if seg_channel not in seg_channels:
            raise ValueError(f"Segmentation channel '{seg_channel}' not found. Available: {seg_channels}")
        seg_idx = seg_channels.index(seg_channel)
        seg_tensor = seg_samples[:, seg_idx, :, :]  # (B, H, W)

        # Convert to list of integer masks (cmasks)
        seg_np = seg_tensor.numpy()
        cmasks_list = [seg_np[i].astype(np.int32) for i in range(seg_np.shape[0])]

        # Poses
        poses_list = poses  # Already (N, 4, 4) numpy array

        num_faces = len(mesh.faces)
        n_views = len(faces_list)
        print(f"  Mesh has {num_faces} faces")
        print(f"  Processing {n_views} views")
        print(f"  Using segmentation channel: {seg_channel}")

        # Run lifting algorithm
        face2label = lift_masks_to_3d(
            faces_list=faces_list,
            cmasks_list=cmasks_list,
            norms_list=norms_list,
            poses_list=poses_list,
            face2label_threshold=face2label_threshold,
            connections_threshold=connections_threshold,
        )

        # Build FACE_LABELS output
        label_counts = {}
        for label in face2label.values():
            label_counts[label] = label_counts.get(label, 0) + 1

        face_labels_data = {
            'face2label': face2label,
            'num_faces': num_faces,
            'label_stats': label_counts,
        }

        labeled_count = len(face2label)
        unique_labels = len(set(face2label.values()))
        print(f"  Labeled {labeled_count}/{num_faces} faces ({100*labeled_count/num_faces:.1f}%)")
        print(f"  Found {unique_labels} unique labels")

        return (face_labels_data,)
