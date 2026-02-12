# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Apply Labels To Mesh Node - Colors mesh faces by segment labels.
"""

import numpy as np
import torch
import trimesh
from numpy.random import RandomState
from PIL import Image

from .types import FACE_LABELS


def numpy_to_tensor(arrays: list, normalize: bool = True) -> torch.Tensor:
    """Convert list of numpy arrays to ComfyUI image tensor."""
    tensors = []
    for arr in arrays:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if normalize and arr.max() > 1:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        arr = np.clip(arr, 0, 1)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        tensors.append(arr)
    if not tensors:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    return torch.from_numpy(np.stack(tensors, axis=0))


def pil_to_tensor(images: list) -> torch.Tensor:
    """Convert list of PIL Images to ComfyUI image tensor (B, H, W, C) float32 0-1."""
    tensors = []
    for img in images:
        if isinstance(img, Image.Image):
            arr = np.array(img).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            tensors.append(arr)
    if not tensors:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    return torch.from_numpy(np.stack(tensors, axis=0))


class ApplyLabelsToMesh:
    """
    Applies face labels to a mesh, coloring each segment with a distinct color.

    Also renders a preview of the segmented mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "face_labels": (FACE_LABELS,),
            },
            "optional": {
                "colormap_seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Random seed for segment colors."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("segmented_mesh", "preview_render")
    FUNCTION = "apply_labels"
    CATEGORY = "meshsegmenter/sammesh"

    def apply_labels(
        self,
        mesh: trimesh.Trimesh,
        face_labels: dict,
        colormap_seed: int = 42
    ):
        from ...samesh.utils.mesh import duplicate_verts

        print("ApplyLabelsToMesh: Applying segment colors to mesh...")

        face2label = face_labels['face2label']
        num_faces = len(mesh.faces)

        # Create color palette
        max_label = max(face2label.values()) if face2label else 1
        rng = RandomState(colormap_seed)
        palette = rng.randint(0, 255, (max_label + 1, 3)).astype(np.uint8)
        palette[0] = [0, 0, 0]  # Background/unlabeled is black

        # Duplicate vertices to prevent color interpolation
        mesh_colored = duplicate_verts(mesh.copy())

        # Apply colors to faces
        labeled_count = 0
        for face_idx, label in face2label.items():
            face_idx = int(face_idx)
            if face_idx < num_faces:
                mesh_colored.visual.face_colors[face_idx, :3] = palette[label]
                mesh_colored.visual.face_colors[face_idx, 3] = 255
                if label > 0:
                    labeled_count += 1

        # Store labels as face attribute
        seg_labels = np.zeros(num_faces, dtype=np.int32)
        for face_idx, label in face2label.items():
            idx = int(face_idx)
            if idx < num_faces:
                seg_labels[idx] = label
        mesh_colored.face_attributes['seg'] = seg_labels

        print(f"  Colored {labeled_count} faces with {max_label} segments")

        # Render preview
        preview_images = self._render_preview(mesh_colored)
        preview_tensor = pil_to_tensor(preview_images)

        print(f"  Preview shape: {preview_tensor.shape}")

        return (mesh_colored, preview_tensor)

    def _render_preview(self, mesh: trimesh.Trimesh, resolution: int = 512) -> list:
        """Render 4 preview views of the colored mesh."""
        import os
        os.environ["PYOPENGL_PLATFORM"] = "egl"

        from omegaconf import OmegaConf
        from ...samesh.renderer.renderer import Renderer

        # Create renderer
        config = OmegaConf.create({
            "target_dim": [resolution, resolution],
        })

        renderer = Renderer(config)
        renderer.set_object(mesh)
        renderer.set_camera()

        # Define 4 camera poses (front, right, back, left)
        poses = []
        radius = 3.0
        for angle in [0, 90, 180, 270]:
            rad = np.radians(angle)
            x = radius * np.sin(rad)
            z = radius * np.cos(rad)
            pose = np.array([
                [np.cos(rad), 0, np.sin(rad), x],
                [0, 1, 0, 0],
                [-np.sin(rad), 0, np.cos(rad), z],
                [0, 0, 0, 1]
            ])
            poses.append(pose)

        # Render each view
        images = []
        for pose in poses:
            outputs = renderer.render(pose, uv_map=True)
            images.append(Image.fromarray(outputs['matte']))

        return images
