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

from .types import CAMERA_POSES

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
                "compute_curvature": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Compute dihedral curvature (max edge angle per face). Sharp edges = high value."
                }),
                "compute_feature_edges": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Render feature edges (sharp edges above angle threshold)."
                }),
                "feature_edge_angle": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 90.0,
                    "step": 5.0,
                    "tooltip": "Angle threshold in degrees for feature edge detection."
                }),
            }
        }

    RETURN_TYPES = ("MULTIBAND_IMAGE", "CAMERA_POSES")
    RETURN_NAMES = ("renders", "poses")
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
        sdf_cone_amplitude: int = 120,
        compute_curvature: bool = False,
        compute_feature_edges: bool = False,
        feature_edge_angle: float = 30.0
    ):
        from ...samesh.renderer.renderer import Renderer, render_multiview, colormap_norms

        print(f"MultiViewRenderer: Starting render...")
        print(f"  Resolution: {render_resolution}")
        print(f"  Camera method: {camera_method}")
        print(f"  Camera radius: {camera_radius}")
        print(f"  Compute SDF: {compute_sdf}")
        print(f"  Compute curvature: {compute_curvature}")
        print(f"  Compute feature edges: {compute_feature_edges} (angle={feature_edge_angle}°)")

        render_dim = int(render_resolution)

        # Copy mesh to avoid modifying original
        mesh = mesh.copy()

        # Fix normals to ensure consistent orientation
        mesh.fix_normals()

        # Normalize mesh if requested
        if normalize_mesh:
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
        renderer.set_object(mesh, smooth=True)
        renderer.set_camera()

        # Render multiview
        # Use pyrender's smooth-rendered normals (not trimesh vertex normal interpolation)
        renderer_args = {"interpolate_norms": False}
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

        # 3. SDF (optional) - single channel, normalized 0-1, smoothly interpolated
        if compute_sdf:
            sdf_per_face = self._compute_sdf_per_face(mesh, sdf_rays, sdf_cone_amplitude)
            sdf_tensor = self._sdf_to_pixels_smooth(mesh, sdf_per_face, renders['faces'], renders['bcent'])
        else:
            # Empty placeholder - single channel
            sdf_tensor = torch.zeros((n_views, render_dim, render_dim), dtype=torch.float32)

        # 4. Curvature (optional) - dihedral angle based, single channel, smoothly interpolated
        if compute_curvature:
            curvature_per_face = self._compute_dihedral_curvature_per_face(mesh)
            # normalize=False because curvature is already normalized to [0,1] based on π
            curvature_tensor = self._values_to_pixels_smooth(
                mesh, curvature_per_face, renders['faces'], renders['bcent'], normalize=False
            )
        else:
            curvature_tensor = torch.zeros((n_views, render_dim, render_dim), dtype=torch.float32)

        # 5. Feature edges (optional) - matte with feature edges overlaid (grayscale)
        if compute_feature_edges:
            feature_edge_tensor = self._render_feature_edges(
                mesh, renders['matte'], renders['faces'], renders['poses'],
                feature_edge_angle, render_dim, camera_radius
            )  # Returns (B, H, W) grayscale
        else:
            feature_edge_tensor = torch.zeros((n_views, render_dim, render_dim), dtype=torch.float32)

        # 6. Face mask (single channel MASK: 1=mesh, 0=background)
        face_mask_list = []
        for faces in renders['faces']:
            mask = (faces != -1).astype(np.float32)
            face_mask_list.append(mask)
        face_mask = torch.from_numpy(np.stack(face_mask_list, axis=0))

        # 7. Face ID (integer per pixel, -1 for background, stored as float)
        face_id_list = []
        for faces in renders['faces']:
            face_id_list.append(faces.astype(np.float32))
        face_id = torch.from_numpy(np.stack(face_id_list, axis=0))

        # Combine all outputs into ONE MULTIBAND_IMAGE
        # Channels: normal_x/y/z, matte, sdf, curvature, feature_edges, mask, face_id (9 total)

        # Convert tensors from (B,H,W,C) to (B,C,H,W)
        normals_bchw = normals_tensor.permute(0, 3, 1, 2)  # (B,3,H,W)
        matte_bchw = matte_tensor[:, :, :, 0:1].permute(0, 3, 1, 2)  # (B,1,H,W) - single channel (grayscale)
        sdf_bchw = sdf_tensor.unsqueeze(1)                  # (B,1,H,W) - single channel
        curvature_bchw = curvature_tensor.unsqueeze(1)      # (B,1,H,W) - dihedral curvature
        feature_edges_bchw = feature_edge_tensor.unsqueeze(1)  # (B,1,H,W) - single channel
        mask_bchw = face_mask.unsqueeze(1)                  # (B,1,H,W)
        face_id_bchw = face_id.unsqueeze(1)                 # (B,1,H,W)

        # Concatenate along channel dimension
        all_channels = torch.cat([
            normals_bchw, matte_bchw, sdf_bchw, curvature_bchw,
            feature_edges_bchw, mask_bchw, face_id_bchw
        ], dim=1)

        channel_names = [
            "normal_x", "normal_y", "normal_z",
            "matte",
            "sdf",
            "curvature",
            "feature_edges",
            "mask",
            "face_id"
        ]

        renders_multiband = create_multiband(
            all_channels,
            channel_names=channel_names,
            metadata={
                "source": "multiview_renderer",
                "sdf_computed": compute_sdf,
                "curvature_computed": compute_curvature,
                "feature_edges_computed": compute_feature_edges,
                "feature_edge_angle": feature_edge_angle,
                "n_views": n_views,
                "resolution": render_dim,
            }
        )

        print(f"  Output MULTIBAND_IMAGE shape: {renders_multiband['samples'].shape}")
        print(f"  Channels: {channel_names}")

        # Poses output: (n_views, 4, 4) camera transformation matrices
        poses_array = np.array(renders['poses'])

        return (renders_multiband, poses_array)

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

    def _sdf_to_pixels_smooth(
        self,
        mesh: trimesh.Trimesh,
        sdf_per_face: np.ndarray,
        face_ids: list,
        barycentrics: list,
    ) -> torch.Tensor:
        """
        Convert per-face SDF values to per-pixel with barycentric interpolation.

        Args:
            mesh: The trimesh object
            sdf_per_face: (num_faces,) SDF value per face
            face_ids: List of (H,W) arrays with face indices (-1 = background)
            barycentrics: List of (H,W,3) barycentric coordinate arrays

        Returns:
            Tensor of shape (B, H, W) with smooth interpolated SDF values (0-1)
        """
        # Convert face SDF to vertex SDF (average of adjacent faces)
        num_verts = len(mesh.vertices)
        vertex_sdf = np.zeros(num_verts, dtype=np.float64)
        vertex_count = np.zeros(num_verts, dtype=np.int32)

        for face_idx, face in enumerate(mesh.faces):
            for vert_idx in face:
                vertex_sdf[vert_idx] += sdf_per_face[face_idx]
                vertex_count[vert_idx] += 1

        # Avoid division by zero
        vertex_count = np.maximum(vertex_count, 1)
        vertex_sdf = vertex_sdf / vertex_count

        # Normalize to 0-1
        sdf_min, sdf_max = vertex_sdf.min(), vertex_sdf.max()
        if sdf_max > sdf_min:
            vertex_sdf = (vertex_sdf - sdf_min) / (sdf_max - sdf_min)
        else:
            vertex_sdf = np.zeros_like(vertex_sdf)

        sdf_images = []
        num_faces = len(mesh.faces)

        for face_id, bcent in zip(face_ids, barycentrics):
            h, w = face_id.shape
            sdf_img = np.zeros((h, w), dtype=np.float32)

            # Flatten for vectorized operations
            face_id_flat = face_id.reshape(-1)
            bcent_flat = bcent.reshape(-1, 3)

            # Valid pixels (not background)
            valid_mask = (face_id_flat >= 0) & (face_id_flat < num_faces)

            if valid_mask.any():
                valid_faces = face_id_flat[valid_mask]
                valid_bcent = bcent_flat[valid_mask]

                # Get vertex indices for each face
                vert_indices = mesh.faces[valid_faces]  # (n_valid, 3)

                # Get vertex SDF values
                vert_sdf = vertex_sdf[vert_indices]  # (n_valid, 3)

                # Interpolate using barycentric coordinates
                interpolated = np.sum(vert_sdf * valid_bcent, axis=1)

                # Write back to image
                sdf_img_flat = sdf_img.reshape(-1)
                sdf_img_flat[valid_mask] = interpolated
                sdf_img = sdf_img_flat.reshape(h, w)

            sdf_images.append(sdf_img)

        return torch.from_numpy(np.stack(sdf_images, axis=0))

    def _values_to_pixels_smooth(
        self,
        mesh: trimesh.Trimesh,
        values_per_face: np.ndarray,
        face_ids: list,
        barycentrics: list,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Convert per-face values to per-pixel with barycentric interpolation.

        Args:
            mesh: The trimesh object
            values_per_face: (num_faces,) value per face
            face_ids: List of (H,W) arrays with face indices (-1 = background)
            barycentrics: List of (H,W,3) barycentric coordinate arrays
            normalize: If True, normalize output to 0-1 based on min/max

        Returns:
            Tensor of shape (B, H, W) with smooth interpolated values
        """
        # Convert face values to vertex values (average of adjacent faces)
        num_verts = len(mesh.vertices)
        vertex_vals = np.zeros(num_verts, dtype=np.float64)
        vertex_count = np.zeros(num_verts, dtype=np.int32)

        for face_idx, face in enumerate(mesh.faces):
            for vert_idx in face:
                vertex_vals[vert_idx] += values_per_face[face_idx]
                vertex_count[vert_idx] += 1

        # Avoid division by zero
        vertex_count = np.maximum(vertex_count, 1)
        vertex_vals = vertex_vals / vertex_count

        # Optionally normalize to 0-1
        if normalize:
            v_min, v_max = vertex_vals.min(), vertex_vals.max()
            if v_max > v_min:
                vertex_vals = (vertex_vals - v_min) / (v_max - v_min)
            else:
                vertex_vals = np.zeros_like(vertex_vals)

        images = []
        num_faces = len(mesh.faces)

        for face_id, bcent in zip(face_ids, barycentrics):
            h, w = face_id.shape
            img = np.zeros((h, w), dtype=np.float32)

            # Flatten for vectorized operations
            face_id_flat = face_id.reshape(-1)
            bcent_flat = bcent.reshape(-1, 3)

            # Valid pixels (not background)
            valid_mask = (face_id_flat >= 0) & (face_id_flat < num_faces)

            if valid_mask.any():
                valid_faces = face_id_flat[valid_mask]
                valid_bcent = bcent_flat[valid_mask]

                # Get vertex indices for each face
                vert_indices = mesh.faces[valid_faces]  # (n_valid, 3)

                # Get vertex values
                vert_vals = vertex_vals[vert_indices]  # (n_valid, 3)

                # Interpolate using barycentric coordinates
                interpolated = np.sum(vert_vals * valid_bcent, axis=1)

                # Write back to image
                img_flat = img.reshape(-1)
                img_flat[valid_mask] = interpolated
                img = img_flat.reshape(h, w)

            images.append(img)

        return torch.from_numpy(np.stack(images, axis=0))

    def _compute_dihedral_curvature_per_face(
        self,
        mesh: trimesh.Trimesh,
    ) -> np.ndarray:
        """
        Compute dihedral angle based curvature per face.

        For each face, takes the max dihedral angle of its edges.
        Sharp edges = high curvature, smooth regions = low curvature.

        Returns array of shape (num_faces,) with values in [0, 1].
        """
        print(f"  Computing dihedral curvature...")

        num_faces = len(mesh.faces)
        face_max_dihedral = np.zeros(num_faces, dtype=np.float32)

        # face_adjacency: (n_edges, 2) - pairs of adjacent face indices
        # face_adjacency_angles: (n_edges,) - dihedral angle at each edge (radians)
        adjacency = mesh.face_adjacency
        angles = mesh.face_adjacency_angles

        # For each edge, update both adjacent faces with max angle
        for i, (f1, f2) in enumerate(adjacency):
            angle = angles[i]
            face_max_dihedral[f1] = max(face_max_dihedral[f1], angle)
            face_max_dihedral[f2] = max(face_max_dihedral[f2], angle)

        # Normalize: 0 = flat (0 rad), 1 = sharp (pi rad)
        face_curvature = face_max_dihedral / np.pi

        # Cleanup
        face_curvature = np.nan_to_num(face_curvature, nan=0.0, posinf=1.0, neginf=0.0)
        face_curvature = np.clip(face_curvature, 0.0, 1.0)

        print(f"    Dihedral curvature range: [{face_curvature.min():.4f}, {face_curvature.max():.4f}]")

        return face_curvature

    def _render_feature_edges(
        self,
        mesh: trimesh.Trimesh,
        matte_images: list,
        face_ids: list,
        poses: list,
        angle_threshold: float,
        render_dim: int,
        camera_radius: float
    ) -> torch.Tensor:
        """
        Render mesh with feature edges using PyVista.

        Args:
            mesh: The mesh to render
            matte_images: List of PIL matte images (not used, we render fresh)
            face_ids: List of (H,W) face ID buffers (not used)
            poses: List of camera pose matrices
            angle_threshold: Angle in degrees above which edges are "sharp"
            render_dim: Resolution of output images
            camera_radius: Distance from camera to mesh center

        Returns:
            Tensor of shape (B, H, W) grayscale with feature edges as black lines
        """
        import pyvista as pv

        print(f"  Rendering feature edges with PyVista (threshold={angle_threshold}°)...")

        # Convert trimesh to pyvista
        faces_pv = np.hstack([
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            mesh.faces
        ]).ravel()
        pv_mesh = pv.PolyData(mesh.vertices, faces_pv)

        # Extract feature edges (sharp edges above angle threshold)
        feature_edges = pv_mesh.extract_feature_edges(
            feature_angle=angle_threshold,
            boundary_edges=False,
            manifold_edges=False,
            feature_edges=True
        )
        print(f"    Extracted {feature_edges.n_cells} feature edges")

        # Render for each camera pose
        edge_images = []
        for view_idx, pose in enumerate(poses):
            # Create offscreen plotter
            pl = pv.Plotter(off_screen=True, window_size=[render_dim, render_dim])
            pl.set_background('white')

            # Add mesh surface (white)
            pl.add_mesh(pv_mesh, color='white', lighting=True)

            # Add feature edges on top (black lines)
            if feature_edges.n_cells > 0:
                pl.add_mesh(feature_edges, color='black', line_width=2)

            # Extract camera position and orientation from pose matrix
            # pose is camera-to-world transform
            cam_pos = pose[:3, 3]  # Camera position in world
            cam_forward = -pose[:3, 2]  # Camera looks along -Z in its local frame
            cam_up = pose[:3, 1]  # Camera up is +Y in its local frame

            # Set camera
            focal_point = cam_pos + cam_forward * camera_radius
            pl.camera.position = cam_pos
            pl.camera.focal_point = focal_point
            pl.camera.up = cam_up
            pl.camera.view_angle = 60  # Match pyrender FOV

            # Render
            img = pl.screenshot(return_img=True)
            pl.close()

            # Convert to grayscale (0-1)
            if img.ndim == 3:
                gray = np.mean(img.astype(np.float32), axis=2) / 255.0
            else:
                gray = img.astype(np.float32) / 255.0

            edge_images.append(gray)

        print(f"    Feature edge rendering complete")
        return torch.from_numpy(np.stack(edge_images, axis=0))
