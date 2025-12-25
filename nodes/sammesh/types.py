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


# Type validation helpers

def validate_face_labels(data: dict) -> bool:
    """Validate FACE_LABELS structure."""
    required_keys = ['face2label', 'num_faces']
    return all(k in data for k in required_keys)
