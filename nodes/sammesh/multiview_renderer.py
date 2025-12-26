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
                    "tooltip": "Compute mean curvature per vertex (useful for segmentation)."
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

        # 4. Curvature (optional) - single channel, normalized 0-1
        if compute_curvature:
            curvature_per_face = self._compute_curvature_per_face(mesh)
            curvature_tensor = self._values_to_pixels(curvature_per_face, renders['faces'])
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
        curvature_bchw = curvature_tensor.unsqueeze(1)      # (B,1,H,W) - single channel
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

    def _values_to_pixels(
        self,
        values_per_face: np.ndarray,
        face_ids: list,
    ) -> torch.Tensor:
        """
        Convert per-face values to per-pixel using face_id buffer.

        Args:
            values_per_face: (num_faces,) value per face
            face_ids: List of (H,W) arrays with face indices (-1 = background)

        Returns:
            Tensor of shape (B, H, W) with normalized values (0-1)
        """
        # Normalize values to 0-1
        v_min, v_max = values_per_face.min(), values_per_face.max()
        if v_max > v_min:
            values_normalized = (values_per_face - v_min) / (v_max - v_min)
        else:
            values_normalized = np.zeros_like(values_per_face)

        images = []
        for face_id in face_ids:
            h, w = face_id.shape
            img = np.zeros((h, w), dtype=np.float32)

            # Lookup value for each pixel using face_id
            valid_mask = face_id >= 0
            img[valid_mask] = values_normalized[face_id[valid_mask]]

            images.append(img)

        return torch.from_numpy(np.stack(images, axis=0))

    def _compute_curvature_per_face(
        self,
        mesh: trimesh.Trimesh,
    ) -> np.ndarray:
        """
        Compute mean curvature per face using cotangent Laplacian.

        Uses libigl for fast sparse matrix computation.
        Returns array of shape (num_faces,) with absolute mean curvature.
        """
        print(f"  Computing curvature...")

        try:
            import igl

            verts = mesh.vertices.astype(np.float64)
            faces = mesh.faces.astype(np.int32)

            # Build cotangent Laplacian (sparse matrix)
            L = igl.cotmatrix(verts, faces)

            # Mass matrix for normalization
            M = igl.massmatrix(verts, faces, igl.MASSMATRIX_TYPE_VORONOI)
            M_inv = 1.0 / (M.diagonal() + 1e-10)

            # Mean curvature normal: H_n = M^-1 @ L @ V
            Hn = (M_inv[:, None] * (L @ verts))

            # Mean curvature magnitude (factor of 0.5 from definition)
            vertex_curvature = 0.5 * np.linalg.norm(Hn, axis=1)

        except ImportError:
            print(f"    libigl not available, falling back to trimesh")
            try:
                vertex_curvature = trimesh.curvature.discrete_mean_curvature_measure(
                    mesh, mesh.vertices, radius=mesh.scale / 20
                )
            except Exception as e:
                print(f"    Curvature computation failed: {e}, using zeros")
                vertex_curvature = np.zeros(len(mesh.vertices))
        except Exception as e:
            print(f"    igl curvature failed: {e}, using zeros")
            vertex_curvature = np.zeros(len(mesh.vertices))

        # Handle NaN values
        nan_count = np.isnan(vertex_curvature).sum()
        if nan_count > 0:
            print(f"    Warning: {nan_count} NaN values in vertex curvature, replacing with 0")
            vertex_curvature = np.nan_to_num(vertex_curvature, nan=0.0)

        # Convert vertex curvature to face curvature (average of face vertices)
        face_curvature = np.mean(vertex_curvature[mesh.faces], axis=1)

        # Take absolute value (we care about magnitude, not sign)
        face_curvature = np.abs(face_curvature)

        # Final NaN check
        face_curvature = np.nan_to_num(face_curvature, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"    Curvature range: [{face_curvature.min():.6f}, {face_curvature.max():.6f}]")

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
