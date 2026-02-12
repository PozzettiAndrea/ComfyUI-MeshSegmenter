# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Apply Labels to Quad Mesh Node - Applies face labels to PyVista mesh preserving quad topology.
"""

import numpy as np

from .types import FACE_LABELS, QUAD_MESH_INFO, PYVISTA_MESH


class ApplyLabelsToQuadMesh:
    """
    Applies face labels to the original PyVista mesh (preserving quad topology).

    Takes face labels (indexed by original face IDs) and applies them as
    cell data to the PyVista mesh stored in quad_info.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quad_info": (QUAD_MESH_INFO, {
                    "tooltip": "Quad mesh info from PyVistaLoader"
                }),
                "face_labels": (FACE_LABELS, {
                    "tooltip": "Face labels (indexed by original face IDs)"
                }),
            },
            "optional": {
                "colormap_seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Random seed for generating distinct segment colors"
                }),
            }
        }

    RETURN_TYPES = (PYVISTA_MESH,)
    RETURN_NAMES = ("labeled_mesh",)
    FUNCTION = "apply_labels"
    CATEGORY = "MeshSegmenter/SAMesh"
    DESCRIPTION = "Applies face labels to PyVista mesh preserving quad topology"

    def apply_labels(
        self,
        quad_info,
        face_labels: dict,
        colormap_seed: int = 42
    ):
        """
        Apply face labels to original PyVista mesh.

        Args:
            quad_info: QuadMeshInfo with original PyVista mesh
            face_labels: Dict with 'face2label' mapping
            colormap_seed: Random seed for color generation

        Returns:
            PyVista PolyData with 'labels' and 'colors' cell data
        """
        from ...samesh.utils.quad_mesh import apply_labels_to_pyvista

        print("ApplyLabelsToQuadMesh: Applying labels to PyVista mesh...")

        face2label = face_labels['face2label']

        labeled_mesh = apply_labels_to_pyvista(
            quad_info,
            face2label,
            colormap_seed=colormap_seed
        )

        # Log statistics
        num_labeled = len([l for l in face2label.values() if l > 0])
        num_segments = len(set(face2label.values()))

        print(f"  Original mesh: {quad_info.num_original_faces} faces "
              f"({quad_info.num_quads} quads, {quad_info.num_tris} tris)")
        print(f"  Labeled faces: {num_labeled}")
        print(f"  Segments: {num_segments}")

        return (labeled_mesh,)


class SplitQuadMeshByLabels:
    """
    Splits a labeled PyVista mesh into separate meshes by label.

    Each segment becomes a separate PyVista mesh, preserving quad topology.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "labeled_mesh": (PYVISTA_MESH, {
                    "tooltip": "Labeled PyVista mesh from ApplyLabelsToQuadMesh"
                }),
            },
            "optional": {
                "min_faces": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Minimum faces for a segment to be included"
                }),
            }
        }

    RETURN_TYPES = ("LIST",)  # List of PyVista meshes
    RETURN_NAMES = ("mesh_parts",)
    FUNCTION = "split_mesh"
    CATEGORY = "MeshSegmenter/SAMesh"
    DESCRIPTION = "Splits labeled mesh into separate parts by label"

    def split_mesh(
        self,
        labeled_mesh,
        min_faces: int = 1
    ):
        """
        Split mesh into parts by label.

        Args:
            labeled_mesh: PyVista mesh with 'labels' cell data
            min_faces: Minimum faces for inclusion

        Returns:
            List of PyVista meshes, one per segment
        """
        import pyvista as pv

        print("SplitQuadMeshByLabels: Splitting mesh by labels...")

        if 'labels' not in labeled_mesh.cell_data:
            print("  Warning: No 'labels' in cell data, returning original mesh")
            return ([labeled_mesh],)

        labels = labeled_mesh.cell_data['labels']
        unique_labels = np.unique(labels)

        parts = []
        for label in unique_labels:
            if label == 0:
                continue  # Skip unlabeled/background

            mask = labels == label
            if mask.sum() < min_faces:
                continue

            # Extract cells with this label
            part = labeled_mesh.extract_cells(np.where(mask)[0])
            if part.n_cells > 0:
                parts.append(part)

        print(f"  Created {len(parts)} mesh parts from {len(unique_labels)} labels")

        return (parts,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ApplyLabelsToQuadMesh": ApplyLabelsToQuadMesh,
    "SplitQuadMeshByLabels": SplitQuadMeshByLabels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyLabelsToQuadMesh": "Apply Labels to Quad Mesh",
    "SplitQuadMeshByLabels": "Split Quad Mesh by Labels",
}
