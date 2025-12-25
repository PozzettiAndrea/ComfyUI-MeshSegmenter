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
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Smoothness weight. Higher = smoother boundaries, may merge small segments. Ignored if target_segments > 0."
                }),
                "iterations": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of alpha-expansion iterations."
                }),
                "target_segments": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Target number of segments (0 = disabled). Auto-tunes lambda to achieve this count."
                }),
                "lambda_min": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Minimum lambda for target search range."
                }),
                "lambda_max": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.5,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Maximum lambda for target search range."
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
        lambda_weight: float = 1.0,
        iterations: int = 2,
        target_segments: int = 0,
        lambda_min: float = 0.1,
        lambda_max: float = 5.0,
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
