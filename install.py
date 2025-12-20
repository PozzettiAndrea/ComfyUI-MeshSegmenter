#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
MeshSegmenter Installer

Orchestrates modular install scripts for ComfyUI-MeshSegmenter.
"""

import sys

from install_scripts import (
    install_system_dependencies,
    install_python_dependencies,
    install_samesh,
)


def main():
    """Entry point."""
    print("\n" + "="*60)
    print("ComfyUI-MeshSegmenter: Installation")
    print("="*60 + "\n")
    print("This installer will set up:")
    print("  1. System dependencies (OpenGL/EGL libraries on Linux)")
    print("  2. Python dependencies (trimesh, torch, pyrender, etc.)")
    print("  3. SAMesh library (for SAM-based mesh segmentation)")
    print("")

    results = {
        'system_deps': False,
        'python_deps': False,
        'samesh': False,
    }

    # Install in order
    results['system_deps'] = install_system_dependencies()
    results['python_deps'] = install_python_dependencies()
    results['samesh'] = install_samesh()

    # Print summary
    print("\n" + "="*60)
    print("Installation Summary")
    print("="*60)
    print(f"  System Dependencies: {'+ Success' if results['system_deps'] else 'x Failed'}")
    print(f"  Python Dependencies: {'+ Success' if results['python_deps'] else 'x Failed'}")
    print(f"  SAMesh Library:      {'+ Success' if results['samesh'] else '~ Skipped/Missing'}")
    print("="*60 + "\n")

    if results['python_deps']:
        print("Installation completed successfully!")
        print("You can now use ComfyUI-MeshSegmenter nodes in ComfyUI.")
        print("")
        if not results['samesh']:
            print("Note: SAMesh library is not installed.")
            print("SAM-based segmentation nodes will not be available.")
            print("See README.md for instructions on installing SAMesh.")
        print("")
        sys.exit(0)
    else:
        print("Installation completed with issues.")
        if not results['python_deps']:
            print("\nPython dependencies failed to install. You can:")
            print("  1. Try running: pip install -r requirements.txt")
            print("  2. Check your Python environment and permissions")
        print("")
        sys.exit(1)


if __name__ == "__main__":
    main()
