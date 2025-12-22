# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
MeshSegmenter PreStartup Script
- Generates backend_mappings.json for dynamic widget visibility
"""
import os
import re
import json


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


# Run on import
generate_backend_mappings()
