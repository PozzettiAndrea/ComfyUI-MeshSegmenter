# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
MeshSegmenter PreStartup Script
- Checks and installs EGL dependencies for headless rendering
- Generates backend_mappings.json for dynamic widget visibility
- Copies example 3D assets to ComfyUI input folder on startup
"""
import os
import re
import json
import shutil
import sys
import subprocess


def check_and_install_egl():
    """Check for EGL library and attempt to install if missing (Linux only)."""
    if sys.platform != "linux":
        return True

    # Check if libEGL.so is available (not just NVIDIA-specific ones)
    try:
        result = subprocess.run(["/sbin/ldconfig", "-p"], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'libEGL.so' in line and 'nvidia' not in line.lower():
                return True  # EGL found
    except Exception:
        pass

    # EGL not found - print warning and attempt install
    print("\033[33m[MeshSegmenter] WARNING: EGL library not found\033[0m")
    print("[MeshSegmenter] Attempting to install EGL dependencies...")

    try:
        # Try to install (works if user has passwordless sudo)
        result = subprocess.run(
            ["sudo", "-n", "apt-get", "update"],
            capture_output=True,
            timeout=60
        )
        result = subprocess.run(
            ["sudo", "-n", "apt-get", "install", "-y", "libegl1", "libegl-dev", "libgl1-mesa-glx"],
            capture_output=True,
            timeout=120
        )
        if result.returncode == 0:
            print("\033[32m[MeshSegmenter] EGL libraries installed successfully\033[0m")
            # Update ldconfig cache
            subprocess.run(["sudo", "-n", "ldconfig"], capture_output=True, timeout=30)
            return True
    except Exception:
        pass

    # Installation failed - print manual instructions
    print("\033[31m[MeshSegmenter] ERROR: Could not install EGL libraries automatically\033[0m")
    print("[MeshSegmenter] Please run manually:")
    print("  sudo apt-get update && sudo apt-get install -y libegl1 libegl-dev libgl1-mesa-glx")
    print("[MeshSegmenter] Then restart ComfyUI")
    return False


def generate_backend_mappings():
    """
    Parse Python node files and generate backend_mappings.json for JS.

    Reads 'backends' metadata from INPUT_TYPES in node files and writes
    a JSON file that JavaScript can fetch to show/hide widgets dynamically.
    """
    custom_node_dir = os.path.dirname(os.path.abspath(__file__))

    # Node files to parse: (file_path, node_class, backend_widget_name)
    node_files = [
        ("nodes/partfield/segment_by_features.py", "MeshSegSegmentByFeatures", "backend"),
    ]

    mappings = {}
    backend_widgets = {}

    for rel_path, node_class, backend_widget in node_files:
        file_path = os.path.join(custom_node_dir, rel_path)
        if not os.path.exists(file_path):
            print(f"[MeshSegmenter] Warning: {rel_path} not found, skipping")
            continue

        with open(file_path, 'r') as f:
            source = f.read()

        # Extract param -> backends mapping using regex
        # Matches: "param_name": (TYPE, {..., "backends": ["a", "b"], ...})
        node_mapping = {}
        pattern = r'"(\w+)":\s*\([^)]+\{[^}]*"backends":\s*\[([^\]]+)\]'

        for match in re.finditer(pattern, source, re.DOTALL):
            param_name = match.group(1)
            backends_str = match.group(2)
            backends = [b.strip().strip('"\'') for b in backends_str.split(',')]

            for backend in backends:
                if backend not in node_mapping:
                    node_mapping[backend] = []
                node_mapping[backend].append(param_name)

        if node_mapping:
            mappings[node_class] = node_mapping
            backend_widgets[node_class] = backend_widget
            print(f"[MeshSegmenter] Parsed {node_class}: {len(node_mapping)} backends")

    # Write to web/js/backend_mappings.json
    output = {
        "mappings": mappings,
        "backend_widgets": backend_widgets,
    }

    output_path = os.path.join(custom_node_dir, "web", "js", "backend_mappings.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[MeshSegmenter] Generated backend_mappings.json")


def copy_example_assets():
    """Copy all files and folders from assets/ directory to ComfyUI input/3d directory."""
    try:
        import folder_paths

        input_folder = folder_paths.get_input_directory()
        custom_node_dir = os.path.dirname(os.path.abspath(__file__))

        # Create input/3d subdirectory
        input_3d_folder = os.path.join(input_folder, "3d")
        os.makedirs(input_3d_folder, exist_ok=True)

        # Copy entire assets/ folder structure
        assets_folder = os.path.join(custom_node_dir, "assets")
        if not os.path.exists(assets_folder):
            print(f"[MeshSegmenter] Warning: assets folder not found at {assets_folder}")
            return

        copied_count = 0
        for root, dirs, files in os.walk(assets_folder):
            # Calculate relative path from assets folder
            rel_path = os.path.relpath(root, assets_folder)

            # Create corresponding subdirectory in destination
            if rel_path != '.':
                dest_dir = os.path.join(input_3d_folder, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
            else:
                dest_dir = input_3d_folder

            # Copy files
            for file in files:
                source_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)

                if not os.path.exists(dest_file):
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
                    # Show relative path for clarity
                    rel_dest = os.path.join(rel_path, file) if rel_path != '.' else file
                    print(f"[MeshSegmenter] Copied {rel_dest} to input/3d/")

        if copied_count > 0:
            print(f"[MeshSegmenter] Copied {copied_count} asset(s) to {input_3d_folder}")
        else:
            print(f"[MeshSegmenter] All assets already exist in {input_3d_folder}")

    except Exception as e:
        print(f"[MeshSegmenter] Error copying assets: {e}")


# Run on import
check_and_install_egl()
generate_backend_mappings()
copy_example_assets()
