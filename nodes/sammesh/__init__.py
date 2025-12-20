# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAMesh - SAM-based Mesh Segmentation Nodes

This module provides mesh segmentation using SAM2 (Segment Anything Model 2).
"""

from .model_downloader import SamModelDownloader
from .loader import SamMeshLoader
from .segmenter import SamMeshSegmenter
from .exporter import SamMeshExporter
from .renderer import SamMeshRenderer

NODE_CLASS_MAPPINGS = {
    "MeshSegSamModelDownloader": SamModelDownloader,
    "MeshSegSamMeshLoader": SamMeshLoader,
    "MeshSegSamMeshSegmenter": SamMeshSegmenter,
    "MeshSegSamMeshExporter": SamMeshExporter,
    "MeshSegSamMeshRenderer": SamMeshRenderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshSegSamModelDownloader": "SAM Model Downloader",
    "MeshSegSamMeshLoader": "Load Mesh (SAMesh)",
    "MeshSegSamMeshSegmenter": "Segment Mesh (SAMesh)",
    "MeshSegSamMeshExporter": "Export Segments (SAMesh)",
    "MeshSegSamMeshRenderer": "Render Mesh Views (SAMesh)",
}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'SamModelDownloader',
    'SamMeshLoader',
    'SamMeshSegmenter',
    'SamMeshExporter',
    'SamMeshRenderer',
]
