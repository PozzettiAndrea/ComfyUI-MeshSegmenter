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


def _load_partfield_model(partfield_config):
    """Load PartField model from a serializable config dict.

    Called inside the worker process to avoid IPC serialization issues
    with CUDA tensors and C++ extensions.
    """
    import comfy.model_management
    from .partfield_model_downloader import create_partfield_config

    checkpoint_path = partfield_config["checkpoint_path"]
    precision = partfield_config.get("precision", "auto")

    device = comfy.model_management.get_torch_device()
    device_str = str(device)

    # Resolve dtype
    if precision == "auto":
        if comfy.model_management.should_use_bf16(device):
            dtype = torch.bfloat16
        elif comfy.model_management.should_use_fp16(device):
            dtype = torch.float16
        else:
            dtype = torch.float32
    elif precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"PartField: Loading model from {checkpoint_path} on {device_str} ({dtype})")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
        cfg = checkpoint['hyper_parameters']['cfg']
    else:
        cfg = create_partfield_config()

    from .partfield_lib.model_trainer_pvcnn_only_demo import Model
    model = Model(cfg)

    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device_str)
    model.eval()
    del checkpoint

    print(f"PartField: Model loaded on {device_str}")
    return model, device_str


# Module-level cache so model survives across node executions in the same process
_cached_model = None
_cached_model_ckpt = None


def _get_partfield_model(partfield_config):
    """Get or load the PartField model, caching at module level."""
    global _cached_model, _cached_model_ckpt
    ckpt = partfield_config["checkpoint_path"]
    if _cached_model is None or _cached_model_ckpt != ckpt:
        _cached_model, _cached_device = _load_partfield_model(partfield_config)
        _cached_model_ckpt = ckpt
        return _cached_model, _cached_device
    device_str = str(next(_cached_model.parameters()).device)
    return _cached_model, device_str


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

        # Load model inside worker process (lazy, cached)
        model, device = _get_partfield_model(partfield_model)

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
            from .partfield_lib.model.PVCNN.encoder_pc import sample_triplane_feat

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

        # Store as single 2D array: shape (n_faces, 448)
        output_mesh.face_attributes['features'] = face_features.astype(np.float32)

        print(f"PartFieldFeatureExtractor: Done! Stored features with shape {face_features.shape}")

        return (output_mesh,)


NODE_CLASS_MAPPINGS = {
    "MeshSegPartFieldFeatureExtractor": PartFieldFeatureExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshSegPartFieldFeatureExtractor": "PartField Feature Extractor (MeshSeg)",
}
