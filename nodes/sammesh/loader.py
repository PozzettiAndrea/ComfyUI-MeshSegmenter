# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAMesh Mesh Loader Node - Loads mesh files for segmentation.
"""

import os
from pathlib import Path
import trimesh

try:
    import folder_paths
    input_dir = folder_paths.get_input_directory()
except ImportError:
    input_dir = os.getcwd()

# Try to import samesh loader
try:
    from samesh.data.loaders import read_mesh
    SAMESH_AVAILABLE = True
except ImportError:
    SAMESH_AVAILABLE = False
    read_mesh = None

MESH_EXTENSIONS = ['.glb', '.gltf', '.obj', '.ply', '.stl', '.3mf', '.off']


class SamMeshLoader:
    """
    Loads mesh data for SAMesh segmentation.
    Provides a dropdown list of meshes found in the ComfyUI input directory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = []
        try:
            for f in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, f)):
                    _, ext = os.path.splitext(f)
                    if ext.lower() in MESH_EXTENSIONS:
                        files.append(f)
        except Exception as e:
            print(f"\033[93mWarning [SamMeshLoader]: Could not list input directory: {e}\033[0m")

        return {
            "required": {
                "mesh": (sorted(files) if files else ["No mesh files found"], {
                    "tooltip": "Select a mesh file from the ComfyUI input directory."
                }),
            },
            "optional": {
                "normalize_mesh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Scale and translate mesh to fit within a unit cube centered at origin."
                }),
                "process_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply Trimesh's default processing (remove duplicates, merge vertices, fix winding)."
                }),
            }
        }

    RETURN_TYPES = ("MESH", "STRING",)
    RETURN_NAMES = ("mesh", "mesh_path",)
    FUNCTION = "load_mesh"
    CATEGORY = "meshsegmenter/sammesh"

    def load_mesh(self, mesh: str, normalize_mesh: bool = False, process_mesh: bool = True):
        if mesh == "No mesh files found":
            raise ValueError("No mesh files found in the ComfyUI input directory.")

        # Construct full path
        try:
            mesh_full_path = folder_paths.get_annotated_filepath(mesh)
        except:
            mesh_full_path = os.path.join(input_dir, mesh)

        if not mesh_full_path or not os.path.exists(mesh_full_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_full_path}")

        print(f"SamMeshLoader: Loading mesh from: {mesh_full_path}")
        mesh_file_path = Path(mesh_full_path)

        try:
            # Use samesh loader if available, otherwise use trimesh directly
            if SAMESH_AVAILABLE and read_mesh is not None:
                loaded_mesh = read_mesh(mesh_file_path, norm=normalize_mesh, process=process_mesh)
            else:
                loaded_mesh = trimesh.load(mesh_file_path, process=process_mesh)
                if normalize_mesh:
                    # Simple normalization
                    centroid = loaded_mesh.centroid
                    loaded_mesh.vertices -= centroid
                    scale = loaded_mesh.scale
                    if scale > 0:
                        loaded_mesh.vertices /= scale

            if loaded_mesh is None:
                raise ValueError(f"Failed to load mesh from {mesh_full_path}")

            # Handle Scene objects
            if isinstance(loaded_mesh, trimesh.Scene):
                if len(loaded_mesh.geometry) == 1:
                    loaded_mesh = list(loaded_mesh.geometry.values())[0]
                else:
                    loaded_mesh = trimesh.util.concatenate(list(loaded_mesh.geometry.values()))

            if not isinstance(loaded_mesh, trimesh.Trimesh):
                print(f"Warning: Loaded object is {type(loaded_mesh)}, attempting to convert...")
                loaded_mesh = trimesh.Trimesh(
                    vertices=loaded_mesh.vertices,
                    faces=loaded_mesh.faces,
                    process=process_mesh
                )

            print(f"SamMeshLoader: Mesh loaded. Vertices: {len(loaded_mesh.vertices)}, Faces: {len(loaded_mesh.faces)}")
            absolute_mesh_path = os.path.abspath(mesh_full_path)
            return (loaded_mesh, absolute_mesh_path,)

        except Exception as e:
            print(f"\033[31mError loading mesh {mesh_full_path}: {e}\033[0m")
            raise
