# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
ComfyUI MeshSegmenter - Surface Mesh Segmentation Custom Nodes

This package provides mesh segmentation nodes for ComfyUI including:
- SAM-based mesh segmentation (SAMesh)
- PartField neural feature field segmentation
"""

import sys

# Only run initialization when loaded by ComfyUI, not during pytest
if 'pytest' not in sys.modules:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print("\033[34m[MeshSegmenter]\033[0m \033[92mLoaded\033[0m")
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Set web directory for JavaScript extensions
WEB_DIRECTORY = "./web"

# Export the mappings so ComfyUI can discover the nodes
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
