# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Smooth Labels Node - Smooths face labels by removing small components and filling holes.
"""

import trimesh

from .types import FACE_LABELS
from ...samesh.models.smoothing import (
    build_mesh_graph,
    smooth_labels,
    split_disconnected_components,
    fill_unlabeled_faces,
    compute_label_stats
)


class SmoothLabels:
    """
    Smooths face labels by:
    1. Removing small disconnected components
    2. Filling holes by propagating neighbor labels
    3. Optionally splitting disconnected regions with the same label
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "face_labels": (FACE_LABELS,),
            },
            "optional": {
                "smoothing_iterations": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Number of iterations for hole filling."
                }),
                "size_threshold_pct": ("FLOAT", {
                    "default": 0.025,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.005,
                    "tooltip": "Remove components smaller than this % of largest component."
                }),
                "area_threshold_pct": ("FLOAT", {
                    "default": 0.025,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.005,
                    "tooltip": "Remove components with less than this % of largest area."
                }),
                "split_disconnected": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Split disconnected regions with same label into separate labels."
                }),
            }
        }

    RETURN_TYPES = (FACE_LABELS, "STRING")
    RETURN_NAMES = ("smoothed_labels", "stats")
    FUNCTION = "smooth_labels"
    CATEGORY = "meshsegmenter/sammesh"

    def smooth_labels(
        self,
        mesh: trimesh.Trimesh,
        face_labels: dict,
        smoothing_iterations: int = 64,
        size_threshold_pct: float = 0.025,
        area_threshold_pct: float = 0.025,
        split_disconnected: bool = True
    ):
        print("SmoothLabels: Smoothing face labels...")

        face2label = dict(face_labels['face2label'])
        num_faces = face_labels['num_faces']

        # Build mesh adjacency graph
        mesh_graph = build_mesh_graph(mesh)

        # Smooth labels
        print("  Step 1: Removing small components and filling holes...")
        face2label = smooth_labels(
            face2label,
            mesh,
            mesh_graph,
            threshold_percentage_size=size_threshold_pct,
            threshold_percentage_area=area_threshold_pct,
            smoothing_iterations=smoothing_iterations
        )

        # Fill any remaining unlabeled faces
        face2label = fill_unlabeled_faces(face2label, num_faces, unlabeled_value=0)

        # Split disconnected components
        if split_disconnected:
            print("  Step 2: Splitting disconnected components...")
            face2label = split_disconnected_components(face2label, mesh_graph, num_faces)

        # Compute statistics
        label_stats = compute_label_stats(face2label)
        unique_labels = len(label_stats)
        labeled_count = sum(1 for l in face2label.values() if l > 0)
        unlabeled_count = num_faces - labeled_count

        # Build output
        smoothed_labels = {
            'face2label': face2label,
            'num_faces': num_faces,
            'label_stats': label_stats,
        }

        stats = f"""Smoothing Stats:
  Total faces: {num_faces:,}
  Labeled: {labeled_count:,} ({100*labeled_count/num_faces:.1f}%)
  Unlabeled: {unlabeled_count:,} ({100*unlabeled_count/num_faces:.1f}%)
  Segments: {unique_labels}
  Iterations: {smoothing_iterations}
  Size threshold: {size_threshold_pct*100:.1f}%
  Area threshold: {area_threshold_pct*100:.1f}%"""

        print(stats)

        return (smoothed_labels, stats)
