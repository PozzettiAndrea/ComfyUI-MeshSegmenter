# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
ComfyUI MeshSegmenter Nodes
Organized by functional category
"""

# Import all node modules
from . import sammesh
# Future categories:
# from . import region_growing
# from . import spectral
# from . import watershed

# Collect all node class mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(sammesh.NODE_CLASS_MAPPINGS)

# Collect all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(sammesh.NODE_DISPLAY_NAME_MAPPINGS)

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
