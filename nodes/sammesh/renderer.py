# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAMesh Renderer Node - Renders multi-view images of meshes.
"""

import os
import json
import copy
import numpy as np
import torch
from PIL import Image
import trimesh

# Try importing pyrender
try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    pyrender = None


class SamMeshRenderer:
    """
    Renders 4 views (front, right, top, back) of the input mesh
    and combines them into a single image grid.
    """

    @classmethod
    def INPUT_TYPES(cls):
        if not PYRENDER_AVAILABLE:
            return {
                "required": {
                    "error": ("STRING", {"default": "pyrender not installed. Node disabled."})
                }
            }
        return {
            "required": {
                "mesh": ("MESH",),
                "render_resolution": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Resolution for each of the 4 rendered views."
                }),
                "background_color": ("STRING", {
                    "default": "[0.1, 0.1, 0.1, 1.0]",
                    "tooltip": "Background RGBA color as a string list, e.g., '[R, G, B, A]'."
                }),
            },
            "optional": {
                "face2label_path": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Optional path to face2label.json for segment coloring."
                }),
                "force_segment_colors": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true and face2label is provided, color segments distinctly."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_views",)
    FUNCTION = "render_views"
    CATEGORY = "meshsegmenter/sammesh"

    def _parse_color(self, color_str: str, default=(0.1, 0.1, 0.1, 1.0)):
        """Parse color string to RGBA values."""
        try:
            color = json.loads(color_str)
            if isinstance(color, list) and len(color) == 4:
                if any(x > 1.0 for x in color):
                    return [max(0.0, min(255.0, x)) / 255.0 for x in color]
                else:
                    return [max(0.0, min(1.0, x)) for x in color]
        except Exception:
            pass
        return list(default)

    def render_views(
        self,
        mesh: trimesh.Trimesh,
        render_resolution: int,
        background_color: str,
        face2label_path: str = "",
        force_segment_colors: bool = True
    ):
        if not PYRENDER_AVAILABLE:
            raise ImportError("pyrender is required for SamMeshRenderer but was not found.")

        bg_color = self._parse_color(background_color)
        working_mesh = mesh
        is_colored_copy = False

        # Apply segment colors if requested
        if force_segment_colors and face2label_path and os.path.exists(face2label_path):
            print(f"SamMeshRenderer: Applying segment colors from {face2label_path}")
            try:
                mesh_copy = copy.deepcopy(mesh)

                with open(face2label_path, 'r') as f:
                    face2label_data = json.load(f)

                face_labels_map = {int(k): int(v) for k, v in face2label_data.items()}

                if mesh_copy.faces.shape[0] > 0:
                    unique_labels = sorted(set(face_labels_map.values()))

                    if unique_labels:
                        colors_for_labels = [trimesh.visual.random_color() for _ in range(len(unique_labels))]
                        label_to_color = {label: colors_for_labels[i] for i, label in enumerate(unique_labels)}

                        default_color = np.array([128, 128, 128, 255], dtype=np.uint8)
                        new_face_colors = np.tile(default_color, (len(mesh_copy.faces), 1))

                        for face_idx, label_val in face_labels_map.items():
                            if face_idx < len(mesh_copy.faces):
                                color = label_to_color.get(label_val, default_color)
                                new_face_colors[face_idx] = color

                        if not isinstance(mesh_copy.visual, trimesh.visual.ColorVisuals):
                            mesh_copy.visual = trimesh.visual.ColorVisuals()

                        mesh_copy.visual.face_colors = new_face_colors
                        working_mesh = mesh_copy
                        is_colored_copy = True
                        print(f"SamMeshRenderer: Applied segment colors to {len(face_labels_map)} faces.")

            except Exception as e:
                print(f"SamMeshRenderer: Warning - Failed to apply segment colors: {e}")

        # Create pyrender mesh
        render_mesh = pyrender.Mesh.from_trimesh(working_mesh, smooth=not is_colored_copy)

        # Set up scene
        scene = pyrender.Scene(bg_color=bg_color[:3], ambient_light=[0.2, 0.2, 0.2])
        scene.add(render_mesh, pose=np.eye(4))

        # Camera setup
        bounds = mesh.bounds
        if bounds is None:
            center = np.array([0, 0, 0])
            distance = 2.0
        else:
            center = mesh.centroid
            scale = np.max(bounds[1] - bounds[0])
            distance = scale * 1.5

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
        camera_node = scene.add(camera, pose=np.eye(4))

        def look_at(eye, target, up):
            """Create camera-to-world transform matrix."""
            forward = np.subtract(target, eye)
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            if np.linalg.norm(right) < 1e-6:
                right = np.cross(forward, [0, 1, 0] if abs(forward[1]) < 0.99 else [1, 0, 0])
            right = right / np.linalg.norm(right)
            new_up = np.cross(right, forward)
            new_up = new_up / np.linalg.norm(new_up)

            cam_to_world = np.eye(4)
            cam_to_world[0, :3] = right
            cam_to_world[1, :3] = new_up
            cam_to_world[2, :3] = -forward
            cam_to_world[:3, 3] = eye
            return cam_to_world

        z_up = [0, 0, 1]
        poses = {
            "front": look_at(center + [0, -distance * 1.2, distance * 0.4], center, z_up),
            "right": look_at(center + [distance * 1.2, 0, distance * 0.4], center, z_up),
            "top": look_at(center + [0, 0, distance * 1.5], center, [0, 1, 0]),
            "back": look_at(center + [0, distance * 1.2, distance * 0.4], center, z_up),
        }

        renderer = pyrender.OffscreenRenderer(render_resolution, render_resolution)
        rendered_images = {}
        light_node = None

        for key in ["front", "right", "top", "back"]:
            try:
                if light_node is not None and scene.has_node(light_node):
                    scene.remove_node(light_node)

                directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
                light_node = scene.add(directional_light, pose=poses[key])

                scene.set_pose(camera_node, poses[key])
                color, _ = renderer.render(scene)
                rendered_images[key] = Image.fromarray(color, 'RGB')
            except Exception as e:
                print(f"\033[91mError rendering view {key}: {e}\033[0m")
                rendered_images[key] = Image.new('RGB', (render_resolution, render_resolution), (50, 50, 50))

        # Combine into 2x2 grid
        grid_size = 1024
        img_per_view = grid_size // 2
        final_image = Image.new('RGB', (grid_size, grid_size))

        final_image.paste(rendered_images["front"].resize((img_per_view, img_per_view), Image.LANCZOS), (0, 0))
        final_image.paste(rendered_images["right"].resize((img_per_view, img_per_view), Image.LANCZOS), (img_per_view, 0))
        final_image.paste(rendered_images["top"].resize((img_per_view, img_per_view), Image.LANCZOS), (0, img_per_view))
        final_image.paste(rendered_images["back"].resize((img_per_view, img_per_view), Image.LANCZOS), (img_per_view, img_per_view))

        renderer.delete()

        # Convert to tensor
        image_np = np.array(final_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        return (image_tensor,)
