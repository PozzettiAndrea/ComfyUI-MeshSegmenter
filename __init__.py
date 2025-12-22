# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
ComfyUI MeshSegmenter - Surface Mesh Segmentation Custom Nodes

This package provides mesh segmentation nodes for ComfyUI including:
- SAM-based mesh segmentation (SAMesh)
- PartField neural feature field segmentation
- And more to come...
"""

import sys
import os

# Only run initialization when loaded by ComfyUI, not during pytest
if 'pytest' not in sys.modules:
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Add samesh paths to sys.path if present
    samesh_src_dir = os.path.join(current_dir, "samesh-main", "src")
    samesh_third_party_dir = os.path.join(current_dir, "samesh-main", "third_party", "segment-anything-2")

    if os.path.exists(samesh_src_dir) and samesh_src_dir not in sys.path:
        sys.path.insert(0, samesh_src_dir)
        print(f"[MeshSegmenter] Added {samesh_src_dir} to sys.path")

    if os.path.exists(samesh_third_party_dir) and samesh_third_party_dir not in sys.path:
        sys.path.insert(0, samesh_third_party_dir)
        print(f"[MeshSegmenter] Added {samesh_third_party_dir} to sys.path for sam2 import")

    # Check for SAM2 submodule presence
    sam2_module_dir = os.path.join(samesh_third_party_dir, "sam2")
    if os.path.exists(samesh_src_dir) and not os.path.exists(sam2_module_dir):
        print(f"\033[93m[MeshSegmenter] Warning: SAM2 submodule appears incomplete.\033[0m")
        print(f"\033[93m[MeshSegmenter] Run: cd samesh-main && git submodule update --init --recursive\033[0m")

    # Add partfield paths to sys.path if present
    partfield_src_dir = os.path.join(current_dir, "partfield-src")
    if os.path.exists(partfield_src_dir) and partfield_src_dir not in sys.path:
        sys.path.insert(0, partfield_src_dir)
        print(f"[MeshSegmenter] Added {partfield_src_dir} to sys.path for PartField import")

    # Import nodes
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    print("\033[34m[MeshSegmenter]\033[0m \033[92mLoaded\033[0m")
else:
    # During testing, don't import nodes
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Set web directory for JavaScript extensions
WEB_DIRECTORY = "./web"

# Export the mappings so ComfyUI can discover the nodes
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
