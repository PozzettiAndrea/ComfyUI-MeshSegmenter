# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAMesh - SAM-based Mesh Segmentation Nodes

This module provides modular mesh segmentation using SAM2 (Segment Anything Model 2).
The pipeline is decomposed into separate nodes for maximum visibility and customization:

Pipeline:
1. MultiViewRenderer - Render mesh from multiple angles (normals, matte, SDF, mask, face_id)
2. GenerateMasks - Run SAM on rendered images to generate segmentations
3. Lift2DTo3DLabels - Convert 2D masks to 3D face labels
4. SmoothLabels - Remove small components, fill holes, split disconnected regions
5. GraphCutRepartition - (Optional) Refine boundaries with alpha-expansion graph cuts
6. ApplyLabelsToMesh - Color mesh by segments
"""

# Core nodes - always available
from .model_downloader import SamModelLoader
from .loader import SamMeshLoader
from .exporter import SamMeshExporter
from .renderer import SamMeshRenderer

# Modular pipeline nodes
from .multiview_renderer import MultiViewRenderer
from .generate_masks import GenerateMasks
from .lift_labels import Lift2DTo3DLabels
from .smooth_labels import SmoothLabels
from .graph_cut import GraphCutRepartition
from .apply_labels import ApplyLabelsToMesh

NODE_CLASS_MAPPINGS = {
    # Utility nodes
    "MeshSegSamModelLoader": SamModelLoader,
    "MeshSegSamMeshLoader": SamMeshLoader,
    "MeshSegSamMeshExporter": SamMeshExporter,
    "MeshSegSamMeshRenderer": SamMeshRenderer,

    # Modular pipeline nodes
    "MeshSegMultiViewRenderer": MultiViewRenderer,
    "MeshSegGenerateMasks": GenerateMasks,
    "MeshSegLift2DTo3DLabels": Lift2DTo3DLabels,
    "MeshSegSmoothLabels": SmoothLabels,
    "MeshSegGraphCutRepartition": GraphCutRepartition,
    "MeshSegApplyLabelsToMesh": ApplyLabelsToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Utility nodes
    "MeshSegSamModelLoader": "SAM Model Loader",
    "MeshSegSamMeshLoader": "Load Mesh (SAMesh)",
    "MeshSegSamMeshExporter": "Export Segments (SAMesh)",
    "MeshSegSamMeshRenderer": "Render Mesh Views (SAMesh)",

    # Modular pipeline nodes
    "MeshSegMultiViewRenderer": "Multi-View Renderer",
    "MeshSegGenerateMasks": "Generate Masks (SAM)",
    "MeshSegLift2DTo3DLabels": "Lift 2D to 3D Labels",
    "MeshSegSmoothLabels": "Smooth Labels",
    "MeshSegGraphCutRepartition": "Graph Cut Repartition",
    "MeshSegApplyLabelsToMesh": "Apply Labels to Mesh",
}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    # Utility nodes
    'SamModelLoader',
    'SamMeshLoader',
    'SamMeshExporter',
    'SamMeshRenderer',
    # Modular pipeline nodes
    'MultiViewRenderer',
    'GenerateMasks',
    'Lift2DTo3DLabels',
    'SmoothLabels',
    'GraphCutRepartition',
    'ApplyLabelsToMesh',
]
