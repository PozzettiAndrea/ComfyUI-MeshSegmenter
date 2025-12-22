# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
PartField - Neural Feature Field Mesh Segmentation Nodes

This module provides mesh segmentation using PartField (PVCNN + Triplane Transformer).
"""

from .model_downloader import PartFieldModelDownloader
from .segmenter import PartFieldSegmenter
from .feature_visualizer import PartFieldFeatureVisualizer
from .feature_extractor import PartFieldFeatureExtractor
from .segment_by_features import SegmentMeshByFeatures

NODE_CLASS_MAPPINGS = {
    "MeshSegPartFieldModelDownloader": PartFieldModelDownloader,
    "MeshSegPartFieldSegmenter": PartFieldSegmenter,  # Legacy: combined extract+segment
    "MeshSegPartFieldFeatureViz": PartFieldFeatureVisualizer,
    "MeshSegPartFieldFeatureExtractor": PartFieldFeatureExtractor,
    "MeshSegSegmentByFeatures": SegmentMeshByFeatures,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshSegPartFieldModelDownloader": "PartField Model Downloader",
    "MeshSegPartFieldSegmenter": "Segment Mesh (PartField Legacy)",
    "MeshSegPartFieldFeatureViz": "Visualize Features (PartField)",
    "MeshSegPartFieldFeatureExtractor": "Extract Features (PartField)",
    "MeshSegSegmentByFeatures": "Segment Mesh By Features",
}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'PartFieldModelDownloader',
    'PartFieldSegmenter',
    'PartFieldFeatureVisualizer',
    'PartFieldFeatureExtractor',
    'SegmentMeshByFeatures',
]
