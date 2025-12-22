# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
PartField Segmenter Node - Segments mesh using PartField neural features.
"""

import os
import sys
import torch
import numpy as np
import trimesh
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

try:
    import folder_paths
    output_dir = folder_paths.get_output_directory()
    temp_dir = folder_paths.get_temp_directory()
except ImportError:
    output_dir = os.path.join(os.getcwd(), "output")
    temp_dir = os.path.join(os.getcwd(), "temp")

DEFAULT_OUTPUT_DIR = os.path.join(output_dir, "meshsegmenter")
DEFAULT_CACHE_DIR = os.path.join(temp_dir, "meshsegmenter_cache")

# Add partfield-src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PARTFIELD_SRC_DIR = os.path.join(NODE_DIR, "partfield-src")

if PARTFIELD_SRC_DIR not in sys.path:
    sys.path.insert(0, PARTFIELD_SRC_DIR)


def numpy_to_tensor(arrays: list, normalize=True) -> torch.Tensor:
    """Convert list of numpy arrays to ComfyUI image tensor (B, H, W, C) float32 0-1."""
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


class PartFieldSegmenter:
    """
    Segments a mesh using PartField neural feature fields.
    Uses PVCNN encoder + Triplane Transformer to extract features,
    then clusters them to produce part segmentation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "partfield_model": ("PARTFIELD_MODEL",),
            },
            "optional": {
                "num_clusters": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 50,
                    "tooltip": "Number of segments to produce."
                }),
                "clustering_method": (["agglomerative", "kmeans"], {
                    "default": "agglomerative",
                    "tooltip": "Clustering algorithm. Agglomerative uses mesh connectivity."
                }),
                "connectivity_option": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 2,
                    "tooltip": "Face adjacency: 0=naive chain, 1=face MST, 2=component MST"
                }),
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

    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("segmented_mesh", "feature_pca_viz")
    FUNCTION = "segment_mesh"
    CATEGORY = "meshsegmenter/partfield"

    def segment_mesh(
        self,
        mesh: trimesh.Trimesh,
        partfield_model: dict,
        num_clusters: int = 10,
        clustering_method: str = "agglomerative",
        connectivity_option: int = 1,
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
        cfg = partfield_model['config']
        device = partfield_model['device']

        print(f"PartFieldSegmenter: Processing mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Normalize mesh to [-1, 1] range
        vertices = mesh.vertices.copy()
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.9 / (bbmax - bbmin).max()
        vertices_norm = (vertices - center) * scale

        # Sample points for PVCNN input (100k points)
        print("PartFieldSegmenter: Sampling surface points...")
        pc, _ = trimesh.sample.sample_surface(
            trimesh.Trimesh(vertices=vertices_norm, faces=mesh.faces, process=False),
            100000
        )
        pc = torch.from_numpy(pc).float().unsqueeze(0).to(device)

        # Extract features
        print("PartFieldSegmenter: Extracting features...")
        with torch.no_grad():
            # Run PVCNN encoder
            pc_feat = model.pvcnn(pc, pc)

            # Run triplane transformer
            planes = model.triplane_transformer(pc_feat)

            # Split into SDF and part planes
            sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

            # Sample features on mesh faces
            print("PartFieldSegmenter: Sampling face features...")
            tensor_vertices = torch.from_numpy(vertices_norm).float().to(device)
            tensor_faces = torch.from_numpy(mesh.faces).long().to(device)

            # Sample points on each face
            face_points = sample_points_on_faces(tensor_vertices, tensor_faces, n_points_per_face)
            face_points = face_points.reshape(1, -1, 3)

            # Import triplane sampling function
            from partfield.model.PVCNN.encoder_pc import sample_triplane_feat

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

        print(f"PartFieldSegmenter: Extracted features shape: {face_features.shape}")

        # Normalize features
        norms = np.linalg.norm(face_features, axis=-1, keepdims=True)
        norms[norms == 0] = 1
        face_features_norm = face_features / norms

        # Run clustering
        print(f"PartFieldSegmenter: Running {clustering_method} clustering with {num_clusters} clusters...")

        if clustering_method == "kmeans":
            clustering = KMeans(n_clusters=num_clusters, random_state=capped_seed)
            labels = clustering.fit_predict(face_features_norm)
        else:
            # Agglomerative clustering with mesh connectivity
            from run_part_clustering import (
                construct_face_adjacency_matrix_naive,
                construct_face_adjacency_matrix_facemst,
                construct_face_adjacency_matrix_ccmst
            )

            # Build adjacency matrix
            if connectivity_option == 0:
                adj_matrix = construct_face_adjacency_matrix_naive(mesh.faces)
            elif connectivity_option == 1:
                adj_matrix = construct_face_adjacency_matrix_facemst(mesh.faces, mesh.vertices)
            else:
                adj_matrix = construct_face_adjacency_matrix_ccmst(mesh.faces, mesh.vertices)

            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                connectivity=adj_matrix,
                linkage='average'
            )
            labels = clustering.fit_predict(face_features_norm)

        # Relabel to sequential integers
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels], dtype=np.int32)

        print(f"PartFieldSegmenter: Found {len(unique_labels)} segments")

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

        # Create PCA visualization
        pca_img = create_pca_visualization(face_features)
        pca_tensor = numpy_to_tensor([pca_img], normalize=True)

        print(f"PartFieldSegmenter: Done! Mesh has {len(unique_labels)} segments")

        return (segmented_mesh, pca_tensor)
