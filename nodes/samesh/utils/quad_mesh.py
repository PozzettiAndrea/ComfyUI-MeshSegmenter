# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Quad mesh support utilities.

Provides conversion between PyVista meshes (which support quads/n-gons)
and trimesh (triangles only) while tracking the original face mapping.
This enables quad-level segmentation in the SAMesh pipeline.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv
    import trimesh


@dataclass
class QuadMeshInfo:
    """
    Stores mapping between original PyVista mesh (with quads) and triangulated version.

    Attributes:
        pv_mesh: Original PyVista mesh with quads/n-gons preserved
        tri_to_orig: (F_tri,) array mapping each triangle index to original face index
        face_sizes: (F_orig,) array with number of vertices per original face (3, 4, or more)
        num_original_faces: Total number of faces in original mesh
        num_quads: Number of quad faces (4 vertices)
        num_tris: Number of triangle faces (3 vertices)
        num_ngons: Number of n-gon faces (5+ vertices)
    """
    pv_mesh: 'pv.PolyData'
    tri_to_orig: np.ndarray
    face_sizes: np.ndarray
    num_original_faces: int
    num_quads: int
    num_tris: int
    num_ngons: int = 0


def triangulate_pyvista(pv_mesh: 'pv.PolyData') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangulate PyVista mesh while tracking original face mapping.

    PyVista/VTK uses a flat array format for faces:
        [n1, v0, v1, ..., vn1-1, n2, v0, v1, ..., vn2-1, ...]
    where n is the number of vertices in each face.

    This function converts to triangles using:
    - Triangles (3 verts): kept as-is
    - Quads (4 verts): split into 2 triangles (0-1-2, 0-2-3 diagonal)
    - N-gons (5+ verts): fan triangulation from vertex 0

    Args:
        pv_mesh: PyVista PolyData mesh

    Returns:
        tri_faces: (F_tri, 3) array of triangulated faces
        tri_to_orig: (F_tri,) array mapping tri_idx -> original_face_idx
        face_sizes: (F_orig,) array with number of vertices per original face
    """
    faces = pv_mesh.faces

    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    tri_faces = []
    tri_to_orig = []
    face_sizes = []

    orig_idx = 0
    i = 0
    while i < len(faces):
        n_verts = faces[i]
        verts = faces[i + 1:i + 1 + n_verts]
        face_sizes.append(n_verts)

        if n_verts == 3:
            # Already a triangle
            tri_faces.append(verts.copy())
            tri_to_orig.append(orig_idx)
        elif n_verts == 4:
            # Quad -> 2 triangles using 0-2 diagonal split
            # This is a common convention that works well for most quads
            tri_faces.append([verts[0], verts[1], verts[2]])
            tri_faces.append([verts[0], verts[2], verts[3]])
            tri_to_orig.extend([orig_idx, orig_idx])
        elif n_verts > 4:
            # N-gon -> fan triangulation from vertex 0
            for j in range(1, n_verts - 1):
                tri_faces.append([verts[0], verts[j], verts[j + 1]])
                tri_to_orig.append(orig_idx)
        else:
            # Degenerate face (< 3 vertices) - skip
            print(f"Warning: Skipping degenerate face with {n_verts} vertices")

        orig_idx += 1
        i += n_verts + 1

    return (
        np.array(tri_faces, dtype=np.int64),
        np.array(tri_to_orig, dtype=np.int64),
        np.array(face_sizes, dtype=np.int64)
    )


def remap_face_id_buffer(
    face_id: np.ndarray,
    tri_to_orig: np.ndarray
) -> np.ndarray:
    """
    Remap triangle face IDs to original (quad/n-gon) face IDs in a rendered buffer.

    After rendering, each pixel contains the triangle face index that was rendered.
    This function converts those triangle indices to the original face indices,
    so the lifting algorithm works directly on quad-level IDs.

    Args:
        face_id: (H, W) array of triangle face IDs, -1 for background
        tri_to_orig: (F_tri,) mapping from triangle index to original face index

    Returns:
        (H, W) array of original face IDs, -1 preserved for background
    """
    result = face_id.copy()
    valid = face_id >= 0

    # Ensure indices are within bounds
    valid_indices = face_id[valid].astype(np.int64)
    if len(valid_indices) > 0:
        max_idx = valid_indices.max()
        if max_idx >= len(tri_to_orig):
            print(f"Warning: face_id contains index {max_idx} but tri_to_orig has length {len(tri_to_orig)}")
            # Clamp to valid range
            valid_indices = np.clip(valid_indices, 0, len(tri_to_orig) - 1)

        result[valid] = tri_to_orig[valid_indices]

    return result


def pyvista_to_trimesh(
    pv_mesh: 'pv.PolyData',
    process: bool = False
) -> Tuple['trimesh.Trimesh', QuadMeshInfo]:
    """
    Convert PyVista mesh to trimesh for rendering, preserving quad mapping.

    This is the main entry point for quad mesh support. It:
    1. Triangulates the PyVista mesh
    2. Creates a trimesh for rendering
    3. Stores the original mesh and mapping in QuadMeshInfo

    Args:
        pv_mesh: PyVista PolyData mesh (may contain quads/n-gons)
        process: Whether to process the mesh (merge vertices, etc.)

    Returns:
        tri_mesh: trimesh.Trimesh with only triangles
        quad_info: QuadMeshInfo with mapping back to original faces
    """
    import trimesh

    vertices = np.array(pv_mesh.points)
    tri_faces, tri_to_orig, face_sizes = triangulate_pyvista(pv_mesh)

    tri_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=tri_faces,
        process=process
    )

    quad_info = QuadMeshInfo(
        pv_mesh=pv_mesh,
        tri_to_orig=tri_to_orig,
        face_sizes=face_sizes,
        num_original_faces=len(face_sizes),
        num_quads=int(np.sum(face_sizes == 4)),
        num_tris=int(np.sum(face_sizes == 3)),
        num_ngons=int(np.sum(face_sizes > 4)),
    )

    return tri_mesh, quad_info


def build_quad_mesh_graph(quad_info: QuadMeshInfo) -> dict:
    """
    Build face adjacency graph for original mesh topology (quads have 4 edges).

    This is used for label smoothing. Unlike triangle meshes where each face
    has 3 edges, quads have 4 edges and n-gons have n edges.

    Args:
        quad_info: QuadMeshInfo with original PyVista mesh

    Returns:
        Dict mapping face_idx -> set of adjacent face indices
    """
    pv_mesh = quad_info.pv_mesh
    faces = pv_mesh.faces

    # Build edge -> faces mapping
    edge_to_faces = {}

    orig_idx = 0
    i = 0
    while i < len(faces):
        n_verts = faces[i]
        verts = faces[i + 1:i + 1 + n_verts]

        # Add all edges of this face
        for j in range(n_verts):
            v0 = int(verts[j])
            v1 = int(verts[(j + 1) % n_verts])
            edge = tuple(sorted((v0, v1)))

            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(orig_idx)

        orig_idx += 1
        i += n_verts + 1

    # Build adjacency graph from shared edges
    mesh_graph = {i: set() for i in range(quad_info.num_original_faces)}

    for edge, face_indices in edge_to_faces.items():
        for fi in face_indices:
            for fj in face_indices:
                if fi != fj:
                    mesh_graph[fi].add(fj)

    return mesh_graph


def compute_face_areas(quad_info: QuadMeshInfo) -> np.ndarray:
    """
    Compute face areas for original PyVista mesh (handles quads/n-gons).

    Args:
        quad_info: QuadMeshInfo with original PyVista mesh

    Returns:
        (num_original_faces,) array of face areas
    """
    pv_mesh = quad_info.pv_mesh

    # PyVista can compute cell sizes directly
    sized = pv_mesh.compute_cell_sizes(length=False, area=True, volume=False)
    return np.array(sized.cell_data['Area'])


def apply_labels_to_pyvista(
    quad_info: QuadMeshInfo,
    face2label: dict,
    colormap_seed: int = 42
) -> 'pv.PolyData':
    """
    Apply face labels to original PyVista mesh (preserving quad topology).

    Args:
        quad_info: QuadMeshInfo with original PyVista mesh
        face2label: Dict mapping original_face_idx -> label
        colormap_seed: Random seed for generating distinct colors

    Returns:
        PyVista PolyData with 'labels' and 'colors' arrays
    """
    import pyvista as pv
    from numpy.random import RandomState

    pv_mesh = quad_info.pv_mesh.copy()
    num_faces = quad_info.num_original_faces

    # Create label array
    labels = np.zeros(num_faces, dtype=np.int32)
    for face_idx, label in face2label.items():
        if 0 <= int(face_idx) < num_faces:
            labels[int(face_idx)] = int(label)

    # Generate color palette
    max_label = max(face2label.values()) if face2label else 1
    rng = RandomState(colormap_seed)
    palette = rng.randint(0, 255, (max_label + 1, 3)).astype(np.uint8)
    palette[0] = [0, 0, 0]  # Background/unlabeled is black

    # Create color array (RGBA)
    colors = np.zeros((num_faces, 4), dtype=np.uint8)
    colors[:, 3] = 255  # Full opacity
    for face_idx in range(num_faces):
        label = labels[face_idx]
        colors[face_idx, :3] = palette[min(label, len(palette) - 1)]

    # Add arrays to mesh
    pv_mesh.cell_data['labels'] = labels
    pv_mesh.cell_data['colors'] = colors

    return pv_mesh
