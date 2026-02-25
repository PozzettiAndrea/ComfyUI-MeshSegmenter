# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Common mesh operations for segmentation nodes.
"""

import numpy as np
import trimesh


def normalize_mesh(mesh: trimesh.Trimesh, center: bool = True, scale: bool = True) -> trimesh.Trimesh:
    """
    Normalize a mesh to fit within a unit cube.

    Args:
        mesh: Input trimesh object
        center: If True, center the mesh at origin
        scale: If True, scale to fit in unit cube

    Returns:
        Normalized trimesh object
    """
    mesh_copy = mesh.copy()

    if center:
        centroid = mesh_copy.centroid
        mesh_copy.vertices -= centroid

    if scale:
        bounds = mesh_copy.bounds
        max_extent = np.max(bounds[1] - bounds[0])
        if max_extent > 0:
            mesh_copy.vertices /= max_extent

    return mesh_copy


def colorize_mesh_by_labels(
    mesh: trimesh.Trimesh,
    face_labels: dict,
    colormap: str = "random"
) -> trimesh.Trimesh:
    """
    Apply colors to mesh faces based on segment labels.

    Args:
        mesh: Input trimesh object
        face_labels: Dict mapping face index -> label
        colormap: Color scheme to use ("random" or matplotlib colormap name)

    Returns:
        Colored trimesh object
    """
    mesh_copy = mesh.copy()
    unique_labels = sorted(set(face_labels.values()))
    num_labels = len(unique_labels)

    if colormap == "random":
        colors = [trimesh.visual.random_color() for _ in range(num_labels)]
    else:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        colors = [
            (np.array(cmap(i / max(1, num_labels - 1))[:3]) * 255).astype(np.uint8)
            for i in range(num_labels)
        ]
        colors = [np.append(c, 255) for c in colors]

    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    face_colors = np.zeros((len(mesh_copy.faces), 4), dtype=np.uint8)
    face_colors[:] = [128, 128, 128, 255]  # Default grey

    for face_idx, label in face_labels.items():
        if face_idx < len(mesh_copy.faces):
            face_colors[face_idx] = label_to_color.get(label, [128, 128, 128, 255])

    mesh_copy.visual = trimesh.visual.ColorVisuals(mesh=mesh_copy, face_colors=face_colors)

    return mesh_copy


def extract_mesh_segments(
    mesh: trimesh.Trimesh,
    face_labels: dict,
    preserve_texture: bool = False
) -> list:
    """
    Extract individual mesh segments based on face labels.

    Args:
        mesh: Input trimesh object
        face_labels: Dict mapping face index -> label
        preserve_texture: If True, attempt to preserve UV coordinates and textures

    Returns:
        List of trimesh objects, one per segment
    """
    unique_labels = sorted(set(face_labels.values()))
    segments = []

    has_texture = (
        hasattr(mesh, 'visual') and
        isinstance(mesh.visual, trimesh.visual.TextureVisuals) and
        hasattr(mesh.visual, 'uv') and
        mesh.visual.uv is not None and
        len(mesh.visual.uv) == len(mesh.vertices)
    )

    for label in unique_labels:
        face_indices = [idx for idx, lbl in face_labels.items() if lbl == label]

        if not face_indices:
            continue

        face_indices = np.array(face_indices, dtype=np.int64)
        segment_faces = mesh.faces[face_indices]
        unique_vertex_indices = np.unique(segment_faces)
        segment_vertices = mesh.vertices[unique_vertex_indices]

        # Remap vertex indices
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
        new_faces = np.array([
            [vertex_map[v_idx] for v_idx in face]
            for face in segment_faces
        ], dtype=np.int64)

        segment_mesh = trimesh.Trimesh(
            vertices=segment_vertices,
            faces=new_faces,
            process=False
        )

        # Handle texture if requested and available
        if preserve_texture and has_texture:
            try:
                segment_uvs = mesh.visual.uv[unique_vertex_indices]

                if isinstance(mesh.visual.material, trimesh.visual.material.PBRMaterial):
                    texture_image = getattr(mesh.visual.material, 'baseColorTexture', None)
                    if texture_image is not None:
                        new_material = trimesh.visual.material.PBRMaterial(
                            baseColorTexture=texture_image,
                            metallicFactor=getattr(mesh.visual.material, 'metallicFactor', 0.0),
                            roughnessFactor=getattr(mesh.visual.material, 'roughnessFactor', 0.5)
                        )
                        segment_mesh.visual = trimesh.visual.TextureVisuals(
                            uv=segment_uvs,
                            material=new_material
                        )
            except Exception:
                pass  # Fall through to default visual

        segments.append(segment_mesh)

    return segments
