# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
PyVista Mesh Loader Node - Converts PyVista mesh to trimesh for SAMesh pipeline.

This node accepts a PyVista PolyData mesh (which supports quads and n-gons)
and converts it to a triangulated trimesh while preserving the original
face topology mapping. This enables quad-level segmentation.
"""

import numpy as np

from .types import QUAD_MESH_INFO, PYVISTA_MESH
from ...samesh.utils.quad_mesh import pyvista_to_trimesh, QuadMeshInfo


class PyVistaLoader:
    """
    Converts a PyVista mesh to trimesh for rendering, preserving quad mapping.

    Input: PyVista PolyData (may contain quads, n-gons)
    Output: Triangulated trimesh + QuadMeshInfo for face ID remapping

    The QuadMeshInfo preserves the original mesh and provides:
    - tri_to_orig: maps triangle indices back to original face indices
    - This allows the segmentation pipeline to work on original faces (quads)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pv_mesh": (PYVISTA_MESH, {
                    "tooltip": "PyVista PolyData mesh (supports quads, n-gons)"
                }),
            },
            "optional": {
                "process_mesh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, process the mesh (merge duplicate vertices). "
                               "Set False to preserve exact vertex indices."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", QUAD_MESH_INFO)
    RETURN_NAMES = ("mesh", "quad_info")
    FUNCTION = "load_pyvista"
    CATEGORY = "MeshSegmenter/SAMesh"
    DESCRIPTION = "Converts PyVista mesh to trimesh for SAMesh, preserving quad topology mapping"

    def load_pyvista(
        self,
        pv_mesh,
        process_mesh: bool = False
    ):
        """
        Convert PyVista mesh to trimesh while tracking original face mapping.

        Args:
            pv_mesh: PyVista PolyData mesh
            process_mesh: Whether to process the mesh after conversion

        Returns:
            mesh: trimesh.Trimesh (triangulated)
            quad_info: QuadMeshInfo with mapping back to original faces
        """
        print("PyVistaLoader: Converting PyVista mesh to trimesh...")

        # Convert and get mapping
        tri_mesh, quad_info = pyvista_to_trimesh(pv_mesh, process=process_mesh)

        # Log statistics
        print(f"  Original mesh: {quad_info.num_original_faces} faces "
              f"({quad_info.num_tris} tris, {quad_info.num_quads} quads, {quad_info.num_ngons} n-gons)")
        print(f"  Triangulated mesh: {len(tri_mesh.faces)} triangles")
        print(f"  Vertices: {len(tri_mesh.vertices)}")

        return (tri_mesh, quad_info)


class RemapFaceIds:
    """
    Remaps triangle face IDs to original face IDs using quad_info mapping.

    This node should be inserted after MultiViewRenderer to convert
    triangle-level face IDs to quad-level face IDs before lifting.
    Works with MULTIBAND_IMAGE format from MultiViewRenderer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "renders": ("MULTIBAND_IMAGE", {
                    "tooltip": "Multiband renders from MultiViewRenderer"
                }),
                "quad_info": (QUAD_MESH_INFO, {
                    "tooltip": "Quad mesh info from PyVistaLoader"
                }),
            },
        }

    RETURN_TYPES = ("MULTIBAND_IMAGE", QUAD_MESH_INFO)
    RETURN_NAMES = ("renders", "quad_info")
    FUNCTION = "remap_face_ids"
    CATEGORY = "MeshSegmenter/SAMesh"
    DESCRIPTION = "Remaps triangle face IDs to original (quad) face IDs in MULTIBAND_IMAGE"

    def remap_face_ids(self, renders, quad_info):
        """
        Remap face_id channel in MULTIBAND_IMAGE from triangle indices to original face indices.

        Args:
            renders: MULTIBAND_IMAGE dict with 'samples' tensor and 'channel_names'
            quad_info: QuadMeshInfo with tri_to_orig mapping

        Returns:
            renders: Modified MULTIBAND_IMAGE with remapped face_id channel
            quad_info: Passed through for downstream nodes
        """
        import torch
        from ...samesh.utils.quad_mesh import remap_face_id_buffer

        print("RemapFaceIds: Remapping triangle face IDs to original face IDs...")

        # Find face_id channel index
        channel_names = renders.get('channel_names', [])
        if 'face_id' not in channel_names:
            print("  Warning: No 'face_id' channel in renders, skipping remapping")
            return (renders, quad_info)

        face_id_idx = channel_names.index('face_id')

        # Get samples tensor: (B, C, H, W)
        samples = renders['samples']
        B, C, H, W = samples.shape

        # Extract face_id channel
        face_id_tensor = samples[:, face_id_idx, :, :]  # (B, H, W)

        # Remap each view
        remapped_list = []
        for b in range(B):
            face_id_np = face_id_tensor[b].numpy()
            remapped = remap_face_id_buffer(face_id_np, quad_info.tri_to_orig)
            remapped_list.append(torch.from_numpy(remapped.astype(np.float32)))

        remapped_tensor = torch.stack(remapped_list, dim=0)  # (B, H, W)

        # Create new samples tensor with remapped face_id
        new_samples = samples.clone()
        new_samples[:, face_id_idx, :, :] = remapped_tensor

        # Create new renders dict
        new_renders = dict(renders)
        new_renders['samples'] = new_samples

        # Add quad_info to metadata for downstream nodes
        metadata = dict(renders.get('metadata', {}))
        metadata['quad_info'] = quad_info
        metadata['face_ids_remapped'] = True
        metadata['num_original_faces'] = quad_info.num_original_faces
        new_renders['metadata'] = metadata

        print(f"  Remapped {B} face ID buffers")
        print(f"  Original faces: {quad_info.num_original_faces} "
              f"({quad_info.num_tris} tris, {quad_info.num_quads} quads, {quad_info.num_ngons} n-gons)")
        print(f"  Triangle faces: {len(quad_info.tri_to_orig)}")

        return (new_renders, quad_info)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PyVistaLoader": PyVistaLoader,
    "RemapFaceIds": RemapFaceIds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyVistaLoader": "PyVista Mesh Loader (Quad Support)",
    "RemapFaceIds": "Remap Face IDs (Tri→Quad)",
}
