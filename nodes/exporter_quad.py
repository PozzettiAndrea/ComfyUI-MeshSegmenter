# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Quad Mesh Exporter Node - Exports PyVista meshes preserving quad topology.

OBJ format natively supports quads, so we use it as the primary export format.
"""

import os
import numpy as np

from .types import PYVISTA_MESH, FACE_LABELS


class QuadMeshExporter:
    """
    Exports a PyVista mesh to OBJ format, preserving quad topology.

    OBJ format natively supports quads and n-gons, unlike GLB/GLTF which
    require triangulation. Use this node when you need to preserve the
    original quad structure.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": (PYVISTA_MESH, {
                    "tooltip": "PyVista mesh to export (quads preserved)"
                }),
                "filename": ("STRING", {
                    "default": "segmented_mesh",
                    "tooltip": "Output filename (without extension)"
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Output directory (defaults to ComfyUI output folder)"
                }),
                "include_colors": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include vertex colors from 'colors' cell data"
                }),
                "export_mtl": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Export MTL material file with segment colors"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "export_mesh"
    CATEGORY = "MeshSegmenter/SAMesh"
    OUTPUT_NODE = True
    DESCRIPTION = "Exports PyVista mesh to OBJ format preserving quad topology"

    def export_mesh(
        self,
        mesh,
        filename: str = "segmented_mesh",
        output_dir: str = "",
        include_colors: bool = True,
        export_mtl: bool = True,
    ):
        """
        Export PyVista mesh to OBJ format.

        Args:
            mesh: PyVista PolyData mesh
            filename: Output filename (without extension)
            output_dir: Output directory
            include_colors: Include vertex colors
            export_mtl: Export MTL material file

        Returns:
            Full path to exported file
        """
        import folder_paths

        print("QuadMeshExporter: Exporting mesh to OBJ...")

        # Determine output directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()

        os.makedirs(output_dir, exist_ok=True)

        obj_path = os.path.join(output_dir, f"{filename}.obj")
        mtl_path = os.path.join(output_dir, f"{filename}.mtl")

        # Export OBJ with quads preserved
        self._write_obj(mesh, obj_path, mtl_path if export_mtl else None, include_colors)

        print(f"  Exported to: {obj_path}")
        if export_mtl:
            print(f"  Material file: {mtl_path}")

        return (obj_path,)

    def _write_obj(self, mesh, obj_path: str, mtl_path: str = None, include_colors: bool = True):
        """
        Write PyVista mesh to OBJ format, preserving quads.

        Args:
            mesh: PyVista PolyData mesh
            obj_path: Output OBJ file path
            mtl_path: Optional MTL file path
            include_colors: Include vertex colors
        """
        vertices = np.array(mesh.points)
        faces = mesh.faces

        # Get labels and colors if available
        labels = mesh.cell_data.get('labels', None)
        colors = mesh.cell_data.get('colors', None)

        # Build material palette from labels
        materials = {}
        if labels is not None and mtl_path:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                # Find a face with this label to get its color
                label_faces = np.where(labels == label)[0]
                if len(label_faces) > 0 and colors is not None:
                    color = colors[label_faces[0]][:3] / 255.0
                else:
                    # Generate a color
                    np.random.seed(int(label) * 12345)
                    color = np.random.rand(3)
                materials[int(label)] = color

        # Write MTL file
        if mtl_path and materials:
            self._write_mtl(mtl_path, materials)

        # Write OBJ file
        with open(obj_path, 'w') as f:
            f.write(f"# Exported by ComfyUI-MeshSegmenter\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            if mtl_path:
                mtl_name = os.path.basename(mtl_path)
                f.write(f"mtllib {mtl_name}\n")
            f.write("\n")

            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")

            # Write faces (preserving quads)
            i = 0
            face_idx = 0
            current_material = None

            while i < len(faces):
                n_verts = faces[i]
                verts = faces[i + 1:i + 1 + n_verts]

                # Switch material if needed
                if labels is not None and mtl_path:
                    label = int(labels[face_idx])
                    if label != current_material:
                        f.write(f"usemtl segment_{label}\n")
                        current_material = label

                # Write face (OBJ uses 1-indexed vertices)
                vert_indices = " ".join(str(v + 1) for v in verts)
                f.write(f"f {vert_indices}\n")

                i += n_verts + 1
                face_idx += 1

        print(f"    Wrote {len(vertices)} vertices, {face_idx} faces")

    def _write_mtl(self, mtl_path: str, materials: dict):
        """
        Write MTL material file.

        Args:
            mtl_path: Output MTL file path
            materials: Dict mapping label -> RGB color (0-1)
        """
        with open(mtl_path, 'w') as f:
            f.write("# Material file exported by ComfyUI-MeshSegmenter\n\n")

            for label, color in materials.items():
                f.write(f"newmtl segment_{label}\n")
                f.write(f"Kd {color[0]:.4f} {color[1]:.4f} {color[2]:.4f}\n")
                f.write(f"Ka 0.1 0.1 0.1\n")
                f.write(f"Ks 0.0 0.0 0.0\n")
                f.write(f"Ns 10.0\n")
                f.write(f"d 1.0\n")
                f.write(f"illum 2\n\n")

        print(f"    Wrote {len(materials)} materials")


class ExportQuadMeshParts:
    """
    Exports multiple PyVista mesh parts to separate OBJ files.

    Each part is saved as a separate file with the segment label as suffix.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_parts": ("LIST", {
                    "tooltip": "List of PyVista mesh parts from SplitQuadMeshByLabels"
                }),
                "filename_prefix": ("STRING", {
                    "default": "segment",
                    "tooltip": "Filename prefix for each part"
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Output directory (defaults to ComfyUI output folder)"
                }),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("filepaths",)
    FUNCTION = "export_parts"
    CATEGORY = "MeshSegmenter/SAMesh"
    OUTPUT_NODE = True
    DESCRIPTION = "Exports mesh parts to separate OBJ files"

    def export_parts(
        self,
        mesh_parts: list,
        filename_prefix: str = "segment",
        output_dir: str = "",
    ):
        """
        Export mesh parts to separate OBJ files.

        Args:
            mesh_parts: List of PyVista meshes
            filename_prefix: Filename prefix
            output_dir: Output directory

        Returns:
            List of output file paths
        """
        import folder_paths

        print("ExportQuadMeshParts: Exporting mesh parts...")

        # Determine output directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()

        os.makedirs(output_dir, exist_ok=True)

        filepaths = []
        for i, part in enumerate(mesh_parts):
            obj_path = os.path.join(output_dir, f"{filename_prefix}_{i:03d}.obj")

            # Use PyVista's built-in save (will triangulate if needed, but preserves quads in many cases)
            part.save(obj_path)
            filepaths.append(obj_path)

            print(f"  Exported part {i}: {obj_path} ({part.n_cells} faces)")

        print(f"  Exported {len(filepaths)} parts total")

        return (filepaths,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "QuadMeshExporter": QuadMeshExporter,
    "ExportQuadMeshParts": ExportQuadMeshParts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuadMeshExporter": "Export Quad Mesh (OBJ)",
    "ExportQuadMeshParts": "Export Quad Mesh Parts",
}
