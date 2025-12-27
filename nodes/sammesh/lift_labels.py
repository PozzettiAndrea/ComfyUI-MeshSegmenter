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
                    "tooltip": "Renders from MultiViewRenderer containing face_id (which mesh face each pixel belongs to) and normal channels (for determining if faces point toward camera)."
                }),
                "segmentations": ("MULTIBAND_IMAGE", {
                    "tooltip": "Segmentations from GenerateMasks containing SAM-generated segment masks (seg_00, seg_01, etc.). Each view has its own local segment IDs that will be unified across views."
                }),
                "poses": (CAMERA_POSES, {
                    "tooltip": "Camera poses from MultiViewRenderer. Used to determine which direction each camera is looking for normal-based visibility filtering."
                }),
            },
            "optional": {
                "seg_channel": ("STRING", {
                    "default": "seg_00",
                    "tooltip": "Which segmentation channel to use for lifting (e.g., seg_00, seg_01). Different channels may contain different SAM segmentation results from the GenerateMasks node."
                }),
                "face2label_threshold": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "STEP 1 - Face-to-Label Assignment: Minimum number of pixels a mesh face must occupy within a 2D segment mask to be assigned that label. For each view, we count how many pixels belong to each (face, label) pair. Pairs with fewer pixels than this threshold are discarded as noise. Higher values require stronger evidence before assigning labels, reducing noise but potentially missing small faces. Lower values capture more detail but may introduce false assignments."
                }),
                "connections_threshold": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "STEP 2 - Cross-View Connection Building: Minimum number of shared mesh faces required to establish a connection between two segment labels from different views. When building the match graph, we connect label A from view 1 to label B from view 2 if they both 'see' the same mesh faces. This threshold filters out weak connections. Higher values require more overlap evidence, creating fewer but more reliable connections. Lower values allow more connections but may link unrelated segments."
                }),
                "connections_bin_resolution": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "tooltip": "STEP 3 - Connection Ratio Filtering: Number of histogram bins used to analyze connection strength distribution. After computing connection ratios (what fraction of a label's connections go to each other label), we build a histogram to find a dynamic threshold. More bins give finer granularity for threshold selection. This is an advanced parameter; the default of 100 works well for most cases."
                }),
                "connections_bin_threshold_pct": ("FLOAT", {
                    "default": 0.125,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "STEP 3 - Connection Ratio Filtering: Percentile cutoff for determining the connection ratio threshold. We sort all connection ratios and find the value below which this percentage of connections fall. Connections weaker than this threshold are discarded. Value of 0.125 means we keep the top 87.5% strongest connections. Higher values are more permissive (keep more connections); lower values are stricter (keep only strongest connections)."
                }),
                "counter_lens_threshold_min": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "STEP 3 - Noisy Label Filtering: Minimum threshold for filtering out 'promiscuous' labels that connect to too many other labels. Labels that connect to more segments than max(95th percentile, this value) are considered noise and removed. These are typically background regions or boundary artifacts that falsely match many segments. Higher values are more aggressive at removing potentially noisy labels."
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
        face2label_threshold: int = 16,
        connections_threshold: int = 32,
        connections_bin_resolution: int = 100,
        connections_bin_threshold_pct: float = 0.125,
        counter_lens_threshold_min: int = 16,
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
            connections_bin_resolution=connections_bin_resolution,
            connections_bin_threshold_percentage=connections_bin_threshold_pct,
            counter_lens_threshold_min=counter_lens_threshold_min,
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
