# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Graph Cut Repartition Node - Refines segmentation boundaries using alpha-expansion.
"""

import trimesh

from .types import FACE_LABELS
from ...samesh.models.smoothing import (
    graph_cut_repartition,
    compute_label_stats,
)


class GraphCutRepartition:
    """
    Refines segmentation boundaries using alpha-expansion graph cuts.

    This optimizes an energy function that balances:
    - Data term: keeping faces with their original labels
    - Smoothness term: preferring cuts at sharp dihedral angles

    The result is cleaner segment boundaries that follow natural mesh creases.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "face_labels": (FACE_LABELS,),
            },
            "optional": {
                "lambda_weight": ("FLOAT", {
                    "default": 6.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "SMOOTHNESS WEIGHT (λ): Controls the balance between data fidelity and boundary smoothness in the graph cut energy function. The algorithm minimizes: E = Σ(data_cost) + λ × Σ(smoothness_cost). Data cost is 0 if a face keeps its label, 1 otherwise. Smoothness cost is based on dihedral angles: sharp edges (creases) have LOW cost to cut, flat regions have HIGH cost. Higher λ values produce smoother boundaries that follow mesh creases but may merge small segments. Lower λ values preserve more segments but may have jagged boundaries. Typical range: 1-15. Ignored when target_segments > 0."
                }),
                "iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "ALPHA-EXPANSION ITERATIONS: Number of complete passes through all labels. Each iteration cycles through every unique label and attempts to 'expand' it using min-cut/max-flow optimization. One iteration is usually sufficient for convergence; additional iterations may refine boundaries slightly but with diminishing returns and increased computation time."
                }),
                "target_segments": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "tooltip": "TARGET SEGMENT COUNT: When set > 0, automatically searches for a lambda value that produces approximately this many segments. The algorithm tries multiple lambda values within [lambda_min, lambda_max] and selects the one closest to the target. Set to 0 to use the fixed lambda_weight instead. Useful when you need a specific level of segmentation granularity."
                }),
                "lambda_min": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "LAMBDA SEARCH MINIMUM: Lower bound for lambda when using target_segments mode. Lower lambda values produce more segments (less smoothing). Only used when target_segments > 0."
                }),
                "lambda_max": ("FLOAT", {
                    "default": 15.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 1.0,
                    "tooltip": "LAMBDA SEARCH MAXIMUM: Upper bound for lambda when using target_segments mode. Higher lambda values produce fewer segments (more smoothing). Only used when target_segments > 0."
                }),
                "noise_threshold": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "NOISE THRESHOLD: Minimum number of faces for a segment to be counted when using target_segments mode. Segments with fewer faces than this are considered noise and excluded from the segment count. This prevents tiny artifact segments from inflating the count. Only affects segment counting for lambda search, not the actual segmentation."
                }),
                "lambda_tolerance": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "LAMBDA SEARCH TOLERANCE: Acceptable deviation from target_segments. If the best lambda found produces a segment count within ±tolerance of the target, no warning is issued. Only used when target_segments > 0."
                }),
            }
        }

    RETURN_TYPES = (FACE_LABELS,)
    RETURN_NAMES = ("refined_labels",)
    FUNCTION = "repartition"
    CATEGORY = "meshsegmenter/sammesh"

    def repartition(
        self,
        mesh: trimesh.Trimesh,
        face_labels: dict,
        lambda_weight: float = 6.0,
        iterations: int = 1,
        target_segments: int = 0,
        lambda_min: float = 1.0,
        lambda_max: float = 15.0,
        noise_threshold: int = 10,
        lambda_tolerance: int = 1,
    ):
        print("GraphCutRepartition: Refining boundaries with alpha-expansion...")

        face2label = dict(face_labels['face2label'])
        num_faces = face_labels['num_faces']

        # Run graph cut repartitioning
        target = target_segments if target_segments > 0 else None
        face2label = graph_cut_repartition(
            face2label,
            mesh,
            repartition_lambda=lambda_weight,
            repartition_iterations=iterations,
            target_labels=target,
            lambda_range=(lambda_min, lambda_max),
            tolerance=lambda_tolerance,
            noise_threshold=noise_threshold,
        )

        # Compute stats
        label_stats = compute_label_stats(face2label)
        unique_labels = len(label_stats)

        refined_labels = {
            'face2label': face2label,
            'num_faces': num_faces,
            'label_stats': label_stats,
        }

        print(f"  Refined to {unique_labels} segments")

        return (refined_labels,)
