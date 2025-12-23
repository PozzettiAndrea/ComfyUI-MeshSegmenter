# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
PartField Feature Extractor Node - Extracts 448-dim features per face.
"""

import os
import sys
import torch
import numpy as np
import trimesh



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


class PartFieldFeatureExtractor:
    """
    Extracts PartField neural features (448-dim) for each face of a mesh.
    Output mesh has features stored in face_attributes['features'].
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
                    "tooltip": "Points sampled per face for feature averaging. Lower = faster."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducibility."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh_with_features",)
    FUNCTION = "extract_features"
    CATEGORY = "meshsegmenter/partfield"

    def extract_features(
        self,
        mesh: trimesh.Trimesh,
        partfield_model: dict,
        n_points_per_face: int = 100,
        seed: int = 0
    ):
        import random

        # Set seeds
        capped_seed = seed % (2**32)
        torch.manual_seed(capped_seed)
        np.random.seed(capped_seed)
        random.seed(capped_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(capped_seed)

        # Get model and config
        model = partfield_model['model']
        device = partfield_model['device']

        print(f"PartFieldFeatureExtractor: Processing mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Normalize mesh to [-1, 1] range
        vertices = mesh.vertices.copy()
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.9 / (bbmax - bbmin).max()
        vertices_norm = (vertices - center) * scale

        # Sample points for PVCNN input (100k points)
        print("PartFieldFeatureExtractor: Sampling surface points...")
        pc, _ = trimesh.sample.sample_surface(
            trimesh.Trimesh(vertices=vertices_norm, faces=mesh.faces, process=False),
            100000
        )
        pc = torch.from_numpy(pc).float().unsqueeze(0).to(device)

        # Extract features
        print("PartFieldFeatureExtractor: Extracting features...")
        with torch.no_grad():
            # Run PVCNN encoder
            pc_feat = model.pvcnn(pc, pc)

            # Run triplane transformer
            planes = model.triplane_transformer(pc_feat)

            # Split into SDF and part planes
            sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

            # Sample features on mesh faces
            print("PartFieldFeatureExtractor: Sampling face features...")
            tensor_vertices = torch.from_numpy(vertices_norm).float().to(device)
            tensor_faces = torch.from_numpy(mesh.faces).long().to(device)

            # Sample points on each face
            face_points = sample_points_on_faces(tensor_vertices, tensor_faces, n_points_per_face)
            face_points = face_points.reshape(1, -1, 3)

            # Import triplane sampling function
            from ...partfield.model.PVCNN.encoder_pc import sample_triplane_feat

            # Sample features in batches to avoid OOM
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

                # Reshape and average over points per face
                batch_size = end_idx - start_idx
                if batch_size % n_points_per_face == 0:
                    sampled_feature = sampled_feature.reshape(1, -1, n_points_per_face, sampled_feature.shape[-1])
                    sampled_feature = torch.mean(sampled_feature, dim=2)
                all_samples.append(sampled_feature)

            face_features = torch.cat(all_samples, dim=1)
            face_features = face_features.reshape(-1, 448).cpu().numpy()

        print(f"PartFieldFeatureExtractor: Extracted features shape: {face_features.shape}")

        # Create output mesh with features
        output_mesh = mesh.copy()

        # Store each feature dimension as a separate field
        num_features = face_features.shape[1]
        for i in range(num_features):
            output_mesh.face_attributes[f'features_{i}'] = face_features[:, i].astype(np.float32)

        print(f"PartFieldFeatureExtractor: Done! Stored {num_features} feature channels")

        return (output_mesh,)
