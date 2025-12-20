# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
MeshSegmenter Install Scripts - Modular installation utilities.
"""

from .system_deps import get_platform_info, install_system_dependencies
from .python_deps import install_python_dependencies
from .samesh import install_samesh

__all__ = [
    'get_platform_info',
    'install_system_dependencies',
    'install_python_dependencies',
    'install_samesh',
]
