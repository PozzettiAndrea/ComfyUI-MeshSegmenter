# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAMesh - SAM-based Mesh Segmentation Nodes

This module provides modular mesh segmentation using SAM2 (Segment Anything Model 2).
The pipeline is decomposed into separate nodes for maximum visibility and customization:

Pipeline:
1. MultiViewRenderer - Render mesh from multiple angles (normals, matte, SDF, face_mask)
2. GenerateMasks - Load SAM and run it on rendered images
3. CombineViewMasks - Merge masks from different sources
4. Lift2DTo3DLabels - Convert 2D masks to 3D face labels
5. SmoothLabels - Clean up labels
6. ApplyLabelsToMesh - Color mesh by segments
"""

# Core nodes - always available
from .model_downloader import SamModelDownloader
from .loader import SamMeshLoader
from .exporter import SamMeshExporter
from .renderer import SamMeshRenderer

# New modular pipeline nodes
from .multiview_renderer import MultiViewRenderer
from .generate_masks import GenerateMasks
from .combine_masks import CombineViewMasks
from .lift_labels import Lift2DTo3DLabels
from .smooth_labels import SmoothLabels
from .apply_labels import ApplyLabelsToMesh

NODE_CLASS_MAPPINGS = {
    # Utility nodes
    "MeshSegSamModelDownloader": SamModelDownloader,
    "MeshSegSamMeshLoader": SamMeshLoader,
    "MeshSegSamMeshExporter": SamMeshExporter,
    "MeshSegSamMeshRenderer": SamMeshRenderer,

    # New modular pipeline nodes
    "MeshSegMultiViewRenderer": MultiViewRenderer,
    "MeshSegGenerateMasks": GenerateMasks,
    "MeshSegCombineViewMasks": CombineViewMasks,
    "MeshSegLift2DTo3DLabels": Lift2DTo3DLabels,
    "MeshSegSmoothLabels": SmoothLabels,
    "MeshSegApplyLabelsToMesh": ApplyLabelsToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Utility nodes
    "MeshSegSamModelDownloader": "SAM Model Downloader",
    "MeshSegSamMeshLoader": "Load Mesh (SAMesh)",
    "MeshSegSamMeshExporter": "Export Segments (SAMesh)",
    "MeshSegSamMeshRenderer": "Render Mesh Views (SAMesh)",

    # New modular pipeline nodes
    "MeshSegMultiViewRenderer": "Multi-View Renderer",
    "MeshSegGenerateMasks": "Generate Masks (SAM)",
    "MeshSegCombineViewMasks": "Combine View Masks",
    "MeshSegLift2DTo3DLabels": "Lift 2D to 3D Labels",
    "MeshSegSmoothLabels": "Smooth Labels",
    "MeshSegApplyLabelsToMesh": "Apply Labels to Mesh",
}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    # Utility nodes
    'SamModelDownloader',
    'SamMeshLoader',
    'SamMeshExporter',
    'SamMeshRenderer',
    # New modular pipeline nodes
    'MultiViewRenderer',
    'GenerateMasks',
    'CombineViewMasks',
    'Lift2DTo3DLabels',
    'SmoothLabels',
    'ApplyLabelsToMesh',
]
