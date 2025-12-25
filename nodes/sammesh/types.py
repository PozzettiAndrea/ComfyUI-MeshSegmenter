# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Custom ComfyUI types for SAMesh modular pipeline.

These types carry intermediate data between nodes while allowing
viewable IMAGE outputs for debugging.
"""

# Custom type constants for ComfyUI node type system
MESH_RENDER_DATA = "MESH_RENDER_DATA"
"""
Dict containing raw render buffers for internal pipeline use:
{
    'norms': List[np.ndarray],      # (H,W,3) normals in [-1,1]
    'norms_masked': List[np.ndarray], # normals with back-facing masked
    'faces': List[np.ndarray],      # (H,W) face IDs, -1=background
    'depth': List[np.ndarray],      # (H,W) depth values
    'matte': List[PIL.Image],       # shaded matte renders
    'poses': np.ndarray,            # (N,4,4) camera matrices
    'mesh': trimesh.Trimesh,        # reference to normalized mesh
}
"""

CAMERA_POSES = "CAMERA_POSES"
"""np.ndarray of shape (N, 4, 4) - camera-to-world matrices"""

VIEW_MASKS = "VIEW_MASKS"
"""
Dict containing per-view mask data:
{
    'bmasks': List[np.ndarray],     # (N_masks, H, W) binary masks per view
    'cmasks': List[np.ndarray],     # (H, W) combined label mask per view
    'point_status': List[dict],     # {'valid': [(x,y),...], 'invalid': [(x,y),...]}
}
"""

FACE_LABELS = "FACE_LABELS"
"""
Dict containing face-to-label mapping:
{
    'face2label': Dict[int, int],   # face_idx -> label
    'num_faces': int,               # total faces in mesh
    'label_stats': Dict[int, int],  # label -> face count
}
"""


# Type validation helpers

def validate_mesh_render_data(data: dict) -> bool:
    """Validate MESH_RENDER_DATA structure."""
    required_keys = ['norms', 'faces', 'poses']
    return all(k in data for k in required_keys)


def validate_view_masks(data: dict) -> bool:
    """Validate VIEW_MASKS structure."""
    required_keys = ['bmasks', 'cmasks']
    return all(k in data for k in required_keys)


def validate_face_labels(data: dict) -> bool:
    """Validate FACE_LABELS structure."""
    required_keys = ['face2label', 'num_faces']
    return all(k in data for k in required_keys)
