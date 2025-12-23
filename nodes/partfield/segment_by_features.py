# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
Segment Mesh By Features Node - Clusters mesh faces based on feature vectors.
Supports multiple clustering backends with dynamic parameter visibility.
"""

import os
import sys
import numpy as np
import trimesh
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA



def numpy_to_tensor(arrays: list, normalize=True):
    """Convert list of numpy arrays to ComfyUI image tensor (B, H, W, C) float32 0-1."""
    import torch
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


def create_pca_visualization(features: np.ndarray, size: int = 512) -> np.ndarray:
    """Create a PCA visualization image of features."""
    # Normalize features
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    norms[norms == 0] = 1
    data_scaled = features / norms

    # Apply PCA
    pca = PCA(n_components=3)
    data_reduced = pca.fit_transform(data_scaled)

    # Normalize to 0-1
    data_min = data_reduced.min()
    data_max = data_reduced.max()
    if data_max > data_min:
        data_reduced = (data_reduced - data_min) / (data_max - data_min)
    else:
        data_reduced = np.zeros_like(data_reduced)

    # Create color array
    colors_255 = (data_reduced * 255).astype(np.uint8)

    # Create a simple visualization grid
    n_features = len(features)
    grid_size = int(np.ceil(np.sqrt(n_features)))

    # Create image
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cell_size = size // grid_size

    for i in range(n_features):
        row = i // grid_size
        col = i % grid_size
        y_start = row * cell_size
        x_start = col * cell_size
        if y_start + cell_size <= size and x_start + cell_size <= size:
            img[y_start:y_start + cell_size, x_start:x_start + cell_size] = colors_255[i]

    return img


class SegmentMeshByFeatures:
    """
    Segments a mesh by clustering its face features.
    Supports multiple clustering backends with backend-specific parameters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "backend": (["agglomerative", "kmeans"], {
                    "default": "agglomerative",
                    "tooltip": "Clustering algorithm to use."
                }),
            },
            "optional": {
                # Common parameters
                "num_clusters": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 100,
                    "tooltip": "Number of segments to produce.",
                    "backends": ["agglomerative", "kmeans"],
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducibility.",
                    "backends": ["agglomerative", "kmeans"],
                }),
                # Agglomerative-specific
                "connectivity": (["face_mst", "component_mst", "none"], {
                    "default": "face_mst",
                    "tooltip": "How to compute mesh connectivity. face_mst=face adjacency MST, component_mst=component-aware MST, none=no connectivity constraint.",
                    "backends": ["agglomerative"],
                }),
                "linkage": (["average", "complete", "single", "ward"], {
                    "default": "average",
                    "tooltip": "Linkage criterion for agglomerative clustering.",
                    "backends": ["agglomerative"],
                }),
                # KMeans-specific
                "n_init": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of times KMeans runs with different centroid seeds.",
                    "backends": ["kmeans"],
                }),
                "max_iter": ("INT", {
                    "default": 300,
                    "min": 10,
                    "max": 1000,
                    "tooltip": "Maximum iterations for KMeans convergence.",
                    "backends": ["kmeans"],
                }),
                "keep_features": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep feature vectors on output mesh. Disable to reduce clutter in preview.",
                    "backends": ["agglomerative", "kmeans"],
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("segmented_mesh", "feature_pca_viz")
    FUNCTION = "segment_mesh"
    CATEGORY = "meshsegmenter/segmentation"

    def segment_mesh(
        self,
        mesh: trimesh.Trimesh,
        backend: str = "agglomerative",
        num_clusters: int = 10,
        seed: int = 0,
        # Agglomerative params
        connectivity: str = "face_mst",
        linkage: str = "average",
        # KMeans params
        n_init: int = 10,
        max_iter: int = 300,
        # Output options
        keep_features: bool = False,
    ):
        import random

        # Set seeds
        capped_seed = seed % (2**32)
        np.random.seed(capped_seed)
        random.seed(capped_seed)

        # Check if mesh has features (stored as features_0, features_1, etc.)
        feature_keys = [k for k in mesh.face_attributes.keys() if k.startswith('features_')]
        feature_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically
        if not feature_keys:
            raise ValueError("Mesh does not have 'features_*' in face_attributes. Run PartFieldFeatureExtractor first.")

        # Reconstruct feature array from individual fields
        num_features = len(feature_keys)
        num_faces = len(mesh.faces)
        face_features = np.zeros((num_faces, num_features), dtype=np.float32)
        for i, key in enumerate(feature_keys):
            face_features[:, i] = mesh.face_attributes[key]

        print(f"SegmentMeshByFeatures: Processing mesh with {num_faces} faces, {num_features} feature dimensions")

        # Normalize features
        norms = np.linalg.norm(face_features, axis=-1, keepdims=True)
        norms[norms == 0] = 1
        face_features_norm = face_features / norms

        # Run clustering based on backend
        print(f"SegmentMeshByFeatures: Running {backend} clustering with {num_clusters} clusters...")

        if backend == "kmeans":
            clustering = KMeans(
                n_clusters=num_clusters,
                random_state=capped_seed,
                n_init=n_init,
                max_iter=max_iter
            )
            labels = clustering.fit_predict(face_features_norm)

        elif backend == "agglomerative":
            # Build adjacency matrix if using connectivity
            adj_matrix = None
            if connectivity != "none":
                from run_part_clustering import (
                    construct_face_adjacency_matrix_naive,
                    construct_face_adjacency_matrix_facemst,
                    construct_face_adjacency_matrix_ccmst
                )

                if connectivity == "face_mst":
                    adj_matrix = construct_face_adjacency_matrix_facemst(mesh.faces, mesh.vertices)
                elif connectivity == "component_mst":
                    adj_matrix = construct_face_adjacency_matrix_ccmst(mesh.faces, mesh.vertices)

            # Ward linkage requires euclidean affinity and doesn't work with connectivity
            if linkage == "ward" and adj_matrix is not None:
                print("SegmentMeshByFeatures: Warning - ward linkage ignores connectivity constraint")
                adj_matrix = None

            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                connectivity=adj_matrix,
                linkage=linkage
            )
            labels = clustering.fit_predict(face_features_norm)

        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Relabel to sequential integers
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels], dtype=np.int32)

        print(f"SegmentMeshByFeatures: Found {len(unique_labels)} segments")

        # Color the mesh
        import matplotlib.pyplot as plt
        colormap = plt.cm.get_cmap("tab20", len(unique_labels))
        face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
        for i, label in enumerate(labels):
            color = (np.array(colormap(label % 20)[:3]) * 255).astype(np.uint8)
            face_colors[i] = np.append(color, 255)

        # Create segmented mesh
        segmented_mesh = mesh.copy()
        segmented_mesh.visual = trimesh.visual.ColorVisuals(mesh=segmented_mesh, face_colors=face_colors)
        segmented_mesh.face_attributes['seg'] = labels

        # Remove features from output if not keeping them
        if not keep_features:
            keys_to_remove = [k for k in segmented_mesh.face_attributes.keys() if k.startswith('features_')]
            for key in keys_to_remove:
                del segmented_mesh.face_attributes[key]

        # Create PCA visualization
        pca_img = create_pca_visualization(face_features)
        pca_tensor = numpy_to_tensor([pca_img], normalize=True)

        print(f"SegmentMeshByFeatures: Done! Mesh has {len(unique_labels)} segments")

        return (segmented_mesh, pca_tensor)
