# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Multi-View Renderer Node - Renders mesh from multiple camera angles.
Outputs normals, matte, optional SDF, and face ID mask for SAM processing.
"""

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import torch
import trimesh
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from .types import MESH_RENDER_DATA


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


class MultiViewRenderer:
    """
    Renders a mesh from multiple camera angles.

    Outputs:
    - render_data: Raw buffers for 2D→3D lifting (face IDs, poses, raw normals)
    - normals: RGB normal maps (black background)
    - matte: RGB shaded renders (black background)
    - sdf: RGB SDF renders (black if compute_sdf=False)
    - face_mask: Single channel MASK (1=mesh, 0=background)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
            "optional": {
                "render_resolution": (["1024", "768", "512"], {
                    "default": "1024",
                    "tooltip": "Resolution for each view render."
                }),
                "camera_method": (["icosahedron", "dodecahedron", "cube", "sphere"], {
                    "default": "icosahedron",
                    "tooltip": "Method for camera placement. icosahedron=20 views, dodecahedron=12 views."
                }),
                "camera_radius": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Distance from camera to mesh center."
                }),
                "normalize_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize mesh to [-1, 1] bounds."
                }),
                "compute_sdf": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Compute Shape Diameter Function (expensive but useful for segmentation)."
                }),
                "sdf_rays": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 256,
                    "step": 16,
                    "tooltip": "Number of rays for SDF computation. More = smoother but slower."
                }),
                "sdf_cone_amplitude": ("INT", {
                    "default": 120,
                    "min": 30,
                    "max": 180,
                    "step": 10,
                    "tooltip": "Cone angle in degrees for SDF ray casting."
                }),
            }
        }

    RETURN_TYPES = (MESH_RENDER_DATA, "IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("render_data", "normals", "matte", "sdf", "face_mask")
    FUNCTION = "render_views"
    CATEGORY = "meshsegmenter/sammesh"

    def render_views(
        self,
        mesh: trimesh.Trimesh,
        render_resolution: str = "1024",
        camera_method: str = "icosahedron",
        camera_radius: float = 3.0,
        normalize_mesh: bool = True,
        compute_sdf: bool = False,
        sdf_rays: int = 64,
        sdf_cone_amplitude: int = 120
    ):
        from ...samesh.renderer.renderer import Renderer, render_multiview, colormap_norms

        print(f"MultiViewRenderer: Starting render...")
        print(f"  Resolution: {render_resolution}")
        print(f"  Camera method: {camera_method}")
        print(f"  Camera radius: {camera_radius}")
        print(f"  Compute SDF: {compute_sdf}")

        render_dim = int(render_resolution)

        # Normalize mesh if requested
        if normalize_mesh:
            mesh = mesh.copy()
            centroid = mesh.bounding_box.centroid
            mesh.vertices -= centroid
            max_extent = max(mesh.bounding_box.extents)
            mesh.vertices /= (max_extent / 2) * 1.001  # Scale to [-1, 1] with margin

        # Create renderer config
        config = OmegaConf.create({
            "target_dim": [render_dim, render_dim],
        })

        # Initialize renderer
        renderer = Renderer(config)
        renderer.set_object(mesh)
        renderer.set_camera()

        # Render multiview
        renderer_args = {"interpolate_norms": True}
        sampling_args = {"radius": camera_radius}
        lighting_args = {}

        renders = render_multiview(
            renderer,
            camera_generation_method=camera_method,
            renderer_args=renderer_args,
            sampling_args=sampling_args,
            lighting_args=lighting_args,
            verbose=True,
        )

        n_views = len(renders['faces'])
        print(f"  Generated {n_views} views")

        # Build render_data dict for downstream nodes
        render_data = {
            'norms': renders['norms'],
            'faces': renders['faces'],
            'depth': [r for r in renders['depth']],
            'matte': renders['matte'],
            'poses': np.array(renders['poses']),
            'mesh': mesh,
        }

        # 1. Normals RGB with BLACK background
        norms_images = []
        for norms, faces in zip(renders['norms'], renders['faces']):
            img = colormap_norms(norms)
            arr = np.array(img)
            arr[faces == -1] = 0  # Black background
            norms_images.append(Image.fromarray(arr))
        normals_tensor = pil_to_tensor(norms_images)

        # 2. Matte (already has black background)
        matte_tensor = pil_to_tensor(renders['matte'])

        # 3. SDF (optional)
        if compute_sdf:
            sdf_tensor = self._compute_sdf(
                mesh, renders['poses'], render_dim, sdf_rays, sdf_cone_amplitude
            )
        else:
            # Empty black placeholder
            sdf_tensor = torch.zeros_like(matte_tensor)

        # 4. Face mask (single channel MASK: 1=mesh, 0=background)
        face_mask_list = []
        for faces in renders['faces']:
            mask = (faces != -1).astype(np.float32)
            face_mask_list.append(mask)
        face_mask = torch.from_numpy(np.stack(face_mask_list, axis=0))

        print(f"  Output shapes:")
        print(f"    normals: {normals_tensor.shape}")
        print(f"    matte: {matte_tensor.shape}")
        print(f"    sdf: {sdf_tensor.shape}")
        print(f"    face_mask: {face_mask.shape}")

        return (render_data, normals_tensor, matte_tensor, sdf_tensor, face_mask)

    def _compute_sdf(
        self,
        mesh: trimesh.Trimesh,
        poses: np.ndarray,
        render_dim: int,
        sdf_rays: int,
        sdf_cone_amplitude: int
    ) -> torch.Tensor:
        """Compute SDF and render from same camera poses."""
        from ...samesh.renderer.renderer import Renderer
        from ...samesh.models.shape_diameter_function import (
            prep_mesh_shape_diameter_function,
            shape_diameter_function,
            colormap_shape_diameter_function
        )

        print(f"  Computing SDF (rays={sdf_rays}, cone={sdf_cone_amplitude})...")

        # Prepare mesh for SDF computation
        mesh_sdf = prep_mesh_shape_diameter_function(mesh)

        # Compute SDF values
        sdf_values = shape_diameter_function(
            mesh_sdf,
            rays=sdf_rays,
            cone_amplitude=sdf_cone_amplitude
        )
        print(f"    SDF range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")

        # Colormap mesh with SDF values
        mesh_colored = colormap_shape_diameter_function(mesh_sdf, sdf_values)

        # Create renderer config
        config = OmegaConf.create({
            "target_dim": [render_dim, render_dim],
        })

        # Initialize renderer with SDF-colored mesh
        renderer = Renderer(config)
        renderer.set_object(mesh_colored)
        renderer.set_camera()

        # Render from same camera positions
        sdf_images = []
        for pose in tqdm(poses, desc="    Rendering SDF views"):
            outputs = renderer.render(pose, uv_map=True, interpolate_norms=True)
            sdf_images.append(Image.fromarray(outputs['matte']))

        return pil_to_tensor(sdf_images)
