# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Custom ComfyUI types for SAMesh modular pipeline.

These types carry intermediate data between nodes while allowing
viewable IMAGE outputs for debugging.
"""

# Custom type constants for ComfyUI node type system

CAMERA_POSES = "CAMERA_POSES"
"""np.ndarray of shape (N, 4, 4) - camera-to-world matrices"""

FACE_LABELS = "FACE_LABELS"
"""
Dict containing face-to-label mapping:
{
    'face2label': Dict[int, int],   # face_idx -> label
    'num_faces': int,               # total faces in mesh
    'label_stats': Dict[int, int],  # label -> face count
}
"""

SAM_MODEL = "SAM_MODEL"
"""Loaded SAM2 model instance (Sam2Model object)"""

QUAD_MESH_INFO = "QUAD_MESH_INFO"
"""
QuadMeshInfo dataclass containing:
{
    'pv_mesh': pv.PolyData,           # Original PyVista mesh with quads
    'tri_to_orig': np.ndarray,        # (F_tri,) tri_idx -> original_face_idx
    'face_sizes': np.ndarray,         # (F_orig,) vertices per face (3 or 4)
    'num_original_faces': int,        # Total original faces
    'num_quads': int,                 # Number of quad faces
    'num_tris': int,                  # Number of triangle faces
    'num_ngons': int,                 # Number of n-gon faces (5+)
}
"""

PYVISTA_MESH = "PYVISTA_MESH"
"""PyVista PolyData mesh object (supports quads, n-gons, cell data)"""


# Type validation helpers

def validate_face_labels(data: dict) -> bool:
    """Validate FACE_LABELS structure."""
    required_keys = ['face2label', 'num_faces']
    return all(k in data for k in required_keys)
