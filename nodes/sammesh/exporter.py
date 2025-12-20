# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAMesh Exporter Node - Exports segmented mesh as GLB with separate objects per segment.
"""

import os
import json
import numpy as np
import trimesh

try:
    import folder_paths
    output_base_dir = folder_paths.get_output_directory()
except ImportError:
    output_base_dir = os.path.join(os.getcwd(), "output")


class SamMeshExporter:
    """
    Exports the segmented mesh as a single GLB file containing a Trimesh Scene.
    Each object in the scene corresponds to one segment.
    """

    RETURN_TYPES = ()
    FUNCTION = "export_parts"
    OUTPUT_NODE = True
    CATEGORY = "meshsegmenter/sammesh"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segmented_mesh": ("MESH",),
                "face2label_path": ("STRING", {"forceInput": True}),
                "output_filename": ("STRING", {
                    "default": "segmented_mesh.glb",
                    "tooltip": "Filename for the exported GLB scene."
                }),
            }
        }

    def export_parts(
        self,
        segmented_mesh: trimesh.Trimesh,
        face2label_path: str,
        output_filename: str
    ):
        if not os.path.exists(face2label_path):
            raise FileNotFoundError(f"Face-to-label JSON not found: {face2label_path}")

        if not output_filename.lower().endswith('.glb'):
            output_filename += '.glb'

        final_output_path = os.path.join(output_base_dir, output_filename)
        print(f"SamMeshExporter: Preparing scene: {final_output_path}")

        # Load face labels
        try:
            with open(face2label_path, 'r') as f:
                face2label = json.load(f)
            face_labels = {int(k): int(v) for k, v in face2label.items()}
        except Exception as e:
            raise ValueError(f"Failed to load face2label JSON: {e}")

        unique_labels = sorted(set(face_labels.values()))
        print(f"SamMeshExporter: Found {len(unique_labels)} unique segment labels.")

        base_mesh = segmented_mesh
        mesh_parts = []

        # Check for texture info
        has_texture = (
            hasattr(base_mesh, 'visual') and
            isinstance(base_mesh.visual, trimesh.visual.TextureVisuals) and
            hasattr(base_mesh.visual, 'uv') and
            base_mesh.visual.uv is not None and
            hasattr(base_mesh.visual, 'material') and
            base_mesh.visual.material is not None and
            len(base_mesh.visual.uv) == len(base_mesh.vertices)
        )

        if has_texture:
            print("SamMeshExporter: Base mesh has texture info. Attempting to texture segments.")
        else:
            print("SamMeshExporter: Base mesh lacks texture info. Segments will not be textured.")

        for label in unique_labels:
            face_indices_for_label = [idx for idx, lbl in face_labels.items() if lbl == label]

            if not face_indices_for_label:
                continue

            try:
                face_indices = np.array(face_indices_for_label, dtype=np.int64)
                segment_faces = base_mesh.faces[face_indices]
                unique_vertex_indices = np.unique(segment_faces)
                segment_vertices = base_mesh.vertices[unique_vertex_indices]

                # Remap vertices
                vertex_map = {old: new for new, old in enumerate(unique_vertex_indices)}
                new_faces = np.array([
                    [vertex_map[v] for v in face]
                    for face in segment_faces
                ], dtype=np.int64)

                segment_mesh = trimesh.Trimesh(
                    vertices=segment_vertices,
                    faces=new_faces,
                    process=False
                )

                # Apply texture if available
                if has_texture:
                    try:
                        segment_uvs = base_mesh.visual.uv[unique_vertex_indices]

                        texture_image = None
                        if isinstance(base_mesh.visual.material, trimesh.visual.material.PBRMaterial):
                            texture_image = getattr(base_mesh.visual.material, 'baseColorTexture', None)
                        elif hasattr(base_mesh.visual.material, 'image_texture'):
                            texture_image = base_mesh.visual.material.image_texture

                        if texture_image is not None:
                            new_material = trimesh.visual.material.PBRMaterial(
                                baseColorTexture=texture_image,
                                metallicFactor=getattr(base_mesh.visual.material, 'metallicFactor', 0.0),
                                roughnessFactor=getattr(base_mesh.visual.material, 'roughnessFactor', 0.5)
                            )
                            segment_mesh.visual = trimesh.visual.TextureVisuals(
                                uv=segment_uvs,
                                material=new_material
                            )
                    except Exception as tex_e:
                        print(f"\033[93mSamMeshExporter: Error applying texture to label {label}: {tex_e}\033[0m")
                        segment_mesh.visual = trimesh.visual.ColorVisuals()
                else:
                    # Apply vertex colors if available
                    if (hasattr(base_mesh, 'visual') and
                        hasattr(base_mesh.visual, 'vertex_colors') and
                        len(base_mesh.visual.vertex_colors) == len(base_mesh.vertices)):
                        segment_mesh.visual = trimesh.visual.ColorVisuals(
                            mesh=segment_mesh,
                            vertex_colors=base_mesh.visual.vertex_colors[unique_vertex_indices]
                        )

                mesh_parts.append(segment_mesh)

            except Exception as e:
                print(f"\033[91mSamMeshExporter: Error processing label {label}: {e}\033[0m")
                import traceback
                traceback.print_exc()

        if not mesh_parts:
            print("SamMeshExporter: No mesh parts generated. Skipping export.")
            return {}

        try:
            scene = trimesh.Scene(mesh_parts)
            print(f"SamMeshExporter: Exporting scene with {len(mesh_parts)} segments to {final_output_path}")
            scene.export(final_output_path, file_type='glb')
            print("SamMeshExporter: Export complete.")
        except Exception as e:
            print(f"\033[91mSamMeshExporter: Error exporting scene: {e}\033[0m")
            import traceback
            traceback.print_exc()
            raise

        return {}
