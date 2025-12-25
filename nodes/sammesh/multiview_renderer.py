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

# Import multiband type - use string constant directly for ComfyUI type system
MULTIBAND_IMAGE = "MULTIBAND_IMAGE"


def create_multiband(samples, channel_names=None, metadata=None):
    """Create a MULTIBAND_IMAGE dict from a tensor."""
    import torch
    # Ensure 4D tensor
    if samples.ndim == 3:
        samples = samples.unsqueeze(0)
    if samples.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got {samples.ndim}D")
    # Ensure float32
    if samples.dtype != torch.float32:
        samples = samples.float()
    B, C, H, W = samples.shape
    # Generate default channel names if not provided
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(C)]
    elif len(channel_names) != C:
        raise ValueError(f"channel_names length ({len(channel_names)}) != channels ({C})")
    return {
        'samples': samples,
        'channel_names': list(channel_names),
        'metadata': metadata or {},
    }


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


def image_to_multiband(tensor: torch.Tensor, channel_names: list, metadata: dict = None) -> dict:
    """Convert (B,H,W,C) IMAGE tensor to MULTIBAND_IMAGE dict."""
    # Permute from (B,H,W,C) to (B,C,H,W)
    samples = tensor.permute(0, 3, 1, 2)
    return create_multiband(samples, channel_names, metadata)


def mask_to_multiband(tensor: torch.Tensor, channel_name: str = "mask", metadata: dict = None) -> dict:
    """Convert (B,H,W) MASK tensor to MULTIBAND_IMAGE dict."""
    # Add channel dim: (B,H,W) -> (B,1,H,W)
    samples = tensor.unsqueeze(1)
    return create_multiband(samples, [channel_name], metadata)


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

    RETURN_TYPES = ("MESH_RENDER_DATA", "MULTIBAND_IMAGE")
    RETURN_NAMES = ("render_data", "renders")
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

        # 3. SDF (optional) - single channel, normalized 0-1
        if compute_sdf:
            sdf_per_face = self._compute_sdf_per_face(mesh, sdf_rays, sdf_cone_amplitude)
            sdf_tensor = self._sdf_to_pixels(sdf_per_face, renders['faces'])  # (B, H, W)
        else:
            # Empty placeholder - single channel
            sdf_tensor = torch.zeros((n_views, render_dim, render_dim), dtype=torch.float32)

        # 4. Face mask (single channel MASK: 1=mesh, 0=background)
        face_mask_list = []
        for faces in renders['faces']:
            mask = (faces != -1).astype(np.float32)
            face_mask_list.append(mask)
        face_mask = torch.from_numpy(np.stack(face_mask_list, axis=0))

        # 5. Face ID (integer per pixel, -1 for background, stored as float)
        face_id_list = []
        for faces in renders['faces']:
            face_id_list.append(faces.astype(np.float32))
        face_id = torch.from_numpy(np.stack(face_id_list, axis=0))

        # Combine all outputs into ONE MULTIBAND_IMAGE
        # Channels: normal_x/y/z, matte_r/g/b, sdf, mask, face_id

        # Convert tensors from (B,H,W,C) to (B,C,H,W)
        normals_bchw = normals_tensor.permute(0, 3, 1, 2)  # (B,3,H,W)
        matte_bchw = matte_tensor.permute(0, 3, 1, 2)      # (B,3,H,W)
        sdf_bchw = sdf_tensor.unsqueeze(1)                  # (B,1,H,W) - single channel
        mask_bchw = face_mask.unsqueeze(1)                  # (B,1,H,W)
        face_id_bchw = face_id.unsqueeze(1)                 # (B,1,H,W)

        # Concatenate along channel dimension
        all_channels = torch.cat([normals_bchw, matte_bchw, sdf_bchw, mask_bchw, face_id_bchw], dim=1)

        channel_names = [
            "normal_x", "normal_y", "normal_z",
            "matte_r", "matte_g", "matte_b",
            "sdf",
            "mask",
            "face_id"
        ]

        renders_multiband = create_multiband(
            all_channels,
            channel_names=channel_names,
            metadata={
                "source": "multiview_renderer",
                "sdf_computed": compute_sdf,
                "n_views": n_views,
                "resolution": render_dim,
            }
        )

        print(f"  Output MULTIBAND_IMAGE shape: {renders_multiband['samples'].shape}")
        print(f"  Channels: {channel_names}")

        return (render_data, renders_multiband)

    def _compute_sdf_per_face(
        self,
        mesh: trimesh.Trimesh,
        sdf_rays: int,
        sdf_cone_amplitude: int
    ) -> np.ndarray:
        """Compute SDF values per face. Returns array of shape (num_faces,)."""
        from ...samesh.models.shape_diameter_function import (
            prep_mesh_shape_diameter_function,
            shape_diameter_function,
        )

        print(f"  Computing SDF (rays={sdf_rays}, cone={sdf_cone_amplitude})...")

        # Prepare mesh for SDF computation
        mesh_sdf = prep_mesh_shape_diameter_function(mesh)

        # Compute SDF values per face
        sdf_values = shape_diameter_function(
            mesh_sdf,
            rays=sdf_rays,
            cone_amplitude=sdf_cone_amplitude
        )
        print(f"    SDF range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")

        return sdf_values

    def _sdf_to_pixels(
        self,
        sdf_per_face: np.ndarray,
        face_ids: list,
    ) -> torch.Tensor:
        """
        Convert per-face SDF values to per-pixel using face_id buffer.

        Args:
            sdf_per_face: (num_faces,) SDF value per face
            face_ids: List of (H,W) arrays with face indices (-1 = background)

        Returns:
            Tensor of shape (B, H, W) with normalized SDF values (0-1)
        """
        # Normalize SDF values to 0-1
        sdf_min, sdf_max = sdf_per_face.min(), sdf_per_face.max()
        if sdf_max > sdf_min:
            sdf_normalized = (sdf_per_face - sdf_min) / (sdf_max - sdf_min)
        else:
            sdf_normalized = np.zeros_like(sdf_per_face)

        sdf_images = []
        for face_id in face_ids:
            h, w = face_id.shape
            sdf_img = np.zeros((h, w), dtype=np.float32)

            # Lookup SDF for each pixel using face_id
            valid_mask = face_id >= 0
            sdf_img[valid_mask] = sdf_normalized[face_id[valid_mask]]

            sdf_images.append(sdf_img)

        return torch.from_numpy(np.stack(sdf_images, axis=0))
