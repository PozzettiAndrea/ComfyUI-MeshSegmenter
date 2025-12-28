# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Smooth Labels Node - Smooths face labels by removing small components and filling holes.
"""

import trimesh

from .types import FACE_LABELS, QUAD_MESH_INFO
from ...samesh.models.smoothing import (
    build_mesh_graph,
    smooth_labels,
    split_disconnected_components,
    fill_unlabeled_faces,
    compute_label_stats,
)


class SmoothLabels:
    """
    Smooths face labels by:
    1. Removing small disconnected components
    2. Filling holes by propagating neighbor labels
    3. Optionally splitting disconnected regions with the same label

    Supports quad meshes via optional quad_info input - when provided,
    uses quad-aware adjacency (4 edges per quad instead of 3 per triangle).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "face_labels": (FACE_LABELS,),
            },
            "optional": {
                "quad_info": (QUAD_MESH_INFO, {
                    "tooltip": "Optional quad mesh info for quad-aware smoothing. "
                               "If provided, uses original mesh topology for adjacency."
                }),
                "smoothing_iterations": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 256,
                    "step": 8,
                    "tooltip": "HOLE FILLING ITERATIONS: Number of passes to propagate labels from neighboring faces into unlabeled regions. Each iteration, unlabeled faces adopt the most common label among their already-labeled neighbors. More iterations allow labels to propagate further into large holes. 64 iterations is usually sufficient; increase only if holes remain unfilled."
                }),
                "size_threshold_pct": ("FLOAT", {
                    "default": 0.025,
                    "min": 0.001,
                    "max": 0.5,
                    "step": 0.005,
                    "tooltip": "SIZE THRESHOLD: Remove connected components with fewer faces than this percentage of the largest component. For example, 0.025 (2.5%) removes components that have less than 2.5% as many faces as the largest segment. This cleans up small noise regions. Components must be below BOTH size AND area thresholds to be removed."
                }),
                "area_threshold_pct": ("FLOAT", {
                    "default": 0.025,
                    "min": 0.001,
                    "max": 0.5,
                    "step": 0.005,
                    "tooltip": "AREA THRESHOLD: Remove connected components with less surface area than this percentage of the largest component's area. For example, 0.025 (2.5%) removes components with less than 2.5% of the largest segment's area. This accounts for the fact that some segments may have many tiny faces. Components must be below BOTH size AND area thresholds to be removed."
                }),
                "split_disconnected": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "SPLIT DISCONNECTED REGIONS: When enabled, if two regions have the same label but are not connected on the mesh surface, they will be assigned different labels. This ensures each output segment is a single connected component. Disable to keep disconnected regions unified under the same label (useful if you want to preserve semantic groupings across the mesh)."
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
        quad_info=None,
        smoothing_iterations: int = 64,
        size_threshold_pct: float = 0.025,
        area_threshold_pct: float = 0.025,
        split_disconnected: bool = True
    ):
        print("SmoothLabels: Smoothing face labels...")

        face2label = dict(face_labels['face2label'])
        num_faces = face_labels['num_faces']

        # Build mesh adjacency graph - use quad-aware if quad_info provided
        area_faces = None
        if quad_info is not None:
            from ...samesh.utils.quad_mesh import build_quad_mesh_graph, compute_face_areas
            print("  Using quad-aware adjacency graph")
            mesh_graph = build_quad_mesh_graph(quad_info)
            # Override num_faces with original face count
            num_faces = quad_info.num_original_faces
            # Compute face areas for original (quad) mesh
            area_faces = compute_face_areas(quad_info)
        else:
            mesh_graph = build_mesh_graph(mesh)

        # Smooth labels
        print("  Step 1: Removing small components and filling holes...")
        face2label = smooth_labels(
            face2label,
            mesh,
            mesh_graph,
            threshold_percentage_nfaces=size_threshold_pct,
            threshold_percentage_area=area_threshold_pct,
            smoothing_iterations=smoothing_iterations,
            area_faces=area_faces,
            num_faces=num_faces,
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
