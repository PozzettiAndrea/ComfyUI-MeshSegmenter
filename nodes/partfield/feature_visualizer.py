# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
PartField Feature Visualizer Node - Visualizes PartField features using PCA.
"""

import os
import sys
import torch
import numpy as np
import trimesh
from sklearn.decomposition import PCA

try:
    import folder_paths
    output_dir = folder_paths.get_output_directory()
except ImportError:
    output_dir = os.path.join(os.getcwd(), "output")



def sample_points_on_faces(vertices, faces, n_point_per_face):
    """Sample random barycentric points on mesh faces."""
    n_f = faces.shape[0]
    device = vertices.device
    dtype = vertices.dtype

    u = torch.sqrt(torch.rand((n_f, n_point_per_face, 1), device=device, dtype=dtype))
    v = torch.rand((n_f, n_point_per_face, 1), device=device, dtype=dtype)
    w0 = 1 - u
    w1 = u * (1 - v)
    w2 = u * v

    face_v_0 = torch.index_select(vertices, 0, faces[:, 0].reshape(-1))
    face_v_1 = torch.index_select(vertices, 0, faces[:, 1].reshape(-1))
    face_v_2 = torch.index_select(vertices, 0, faces[:, 2].reshape(-1))
    points = w0 * face_v_0.unsqueeze(dim=1) + w1 * face_v_1.unsqueeze(dim=1) + w2 * face_v_2.unsqueeze(dim=1)
    return points


class PartFieldFeatureVisualizer:
    """
    Visualizes PartField features on a mesh using PCA-based coloring.
    Outputs a mesh colored by feature similarity (similar colors = similar features).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "partfield_model": ("PARTFIELD_MODEL",),
            },
            "optional": {
                "n_points_per_face": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 2000,
                    "tooltip": "Points sampled per face for feature averaging."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("pca_colored_mesh",)
    FUNCTION = "visualize_features"
    CATEGORY = "meshsegmenter/partfield"

    def visualize_features(
        self,
        mesh: trimesh.Trimesh,
        partfield_model: dict,
        n_points_per_face: int = 100
    ):
        # Get model and config
        model = partfield_model['model']
        device = partfield_model['device']

        print(f"PartFieldFeatureVisualizer: Processing mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Normalize mesh to [-1, 1] range
        vertices = mesh.vertices.copy()
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.9 / (bbmax - bbmin).max()
        vertices_norm = (vertices - center) * scale

        # Sample points for PVCNN input
        print("PartFieldFeatureVisualizer: Sampling surface points...")
        pc, _ = trimesh.sample.sample_surface(
            trimesh.Trimesh(vertices=vertices_norm, faces=mesh.faces, process=False),
            100000
        )
        pc = torch.from_numpy(pc).float().unsqueeze(0).to(device)

        # Extract features
        print("PartFieldFeatureVisualizer: Extracting features...")
        with torch.no_grad():
            # Run PVCNN encoder
            pc_feat = model.pvcnn(pc, pc)

            # Run triplane transformer
            planes = model.triplane_transformer(pc_feat)

            # Split into SDF and part planes
            sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

            # Sample features on mesh faces
            tensor_vertices = torch.from_numpy(vertices_norm).float().to(device)
            tensor_faces = torch.from_numpy(mesh.faces).long().to(device)

            # Sample points on each face
            face_points = sample_points_on_faces(tensor_vertices, tensor_faces, n_points_per_face)
            face_points = face_points.reshape(1, -1, 3)

            # Import triplane sampling function
            from ...partfield.model.PVCNN.encoder_pc import sample_triplane_feat

            # Sample features in batches
            n_sample_each = 10000
            n_v = face_points.shape[1]
            n_sample = n_v // n_sample_each + 1
            all_samples = []

            for i_sample in range(n_sample):
                start_idx = i_sample * n_sample_each
                end_idx = min(start_idx + n_sample_each, n_v)
                if start_idx >= n_v:
                    break

                sampled_feature = sample_triplane_feat(
                    part_planes,
                    face_points[:, start_idx:end_idx, :]
                )

                batch_size = end_idx - start_idx
                if batch_size % n_points_per_face == 0:
                    sampled_feature = sampled_feature.reshape(1, -1, n_points_per_face, sampled_feature.shape[-1])
                    sampled_feature = torch.mean(sampled_feature, dim=2)
                all_samples.append(sampled_feature)

            face_features = torch.cat(all_samples, dim=1)
            face_features = face_features.reshape(-1, 448).cpu().numpy()

        print(f"PartFieldFeatureVisualizer: Extracted features shape: {face_features.shape}")

        # Apply PCA for visualization
        print("PartFieldFeatureVisualizer: Computing PCA colors...")
        norms = np.linalg.norm(face_features, axis=-1, keepdims=True)
        norms[norms == 0] = 1
        data_scaled = face_features / norms

        pca = PCA(n_components=3)
        data_reduced = pca.fit_transform(data_scaled)

        # Normalize to 0-1
        data_min = data_reduced.min()
        data_max = data_reduced.max()
        if data_max > data_min:
            data_reduced = (data_reduced - data_min) / (data_max - data_min)
        else:
            data_reduced = np.zeros_like(data_reduced)

        # Create colors
        colors_255 = (data_reduced * 255).astype(np.uint8)
        face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
        for i in range(len(mesh.faces)):
            face_colors[i] = np.append(colors_255[i], 255)

        # Create colored mesh
        pca_mesh = mesh.copy()
        pca_mesh.visual = trimesh.visual.ColorVisuals(mesh=pca_mesh, face_colors=face_colors)

        print("PartFieldFeatureVisualizer: Done!")

        return (pca_mesh,)
