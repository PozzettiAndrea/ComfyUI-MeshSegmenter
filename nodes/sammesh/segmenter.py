# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
SAMesh Segmenter Node - Segments mesh using SAM2.
Runs directly in process (no subprocess).
Outputs intermediate renders for visualization.
"""

import os
import sys
import json
import trimesh
import torch
import numpy as np
import random
from PIL import Image

# Set EGL for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"

try:
    import folder_paths
    output_dir = folder_paths.get_output_directory()
    temp_dir = folder_paths.get_temp_directory()
except ImportError:
    output_dir = os.path.join(os.getcwd(), "output")
    temp_dir = os.path.join(os.getcwd(), "temp")

DEFAULT_OUTPUT_DIR = os.path.join(output_dir, "meshsegmenter")
DEFAULT_CACHE_DIR = os.path.join(temp_dir, "meshsegmenter_cache")

# Add samesh to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SAMESH_SRC_DIR = os.path.join(NODE_DIR, "samesh-main", "src")

if SAMESH_SRC_DIR not in sys.path:
    sys.path.insert(0, SAMESH_SRC_DIR)


def pil_to_tensor(images: list) -> torch.Tensor:
    """Convert list of PIL Images to ComfyUI image tensor (B, H, W, C) float32 0-1."""
    tensors = []
    for img in images:
        if isinstance(img, Image.Image):
            arr = np.array(img).astype(np.float32) / 255.0
            if arr.ndim == 2:  # Grayscale
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[-1] == 4:  # RGBA
                arr = arr[:, :, :3]  # Drop alpha
            tensors.append(arr)
    if not tensors:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    return torch.from_numpy(np.stack(tensors, axis=0))


def numpy_to_tensor(arrays: list, normalize=True) -> torch.Tensor:
    """Convert list of numpy arrays to ComfyUI image tensor (B, H, W, C) float32 0-1."""
    tensors = []
    for arr in arrays:
        if arr.ndim == 2:  # Grayscale or mask
            arr = np.stack([arr, arr, arr], axis=-1)
        if normalize and arr.max() > 1:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        # Clamp to 0-1
        arr = np.clip(arr, 0, 1)
        if arr.shape[-1] == 4:  # RGBA
            arr = arr[:, :, :3]
        tensors.append(arr)
    if not tensors:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    return torch.from_numpy(np.stack(tensors, axis=0))


def colormap_masks(masks: list) -> list:
    """Convert combined masks to colored images for visualization."""
    from numpy.random import RandomState
    colored = []
    for mask in masks:
        # mask is (H, W) with integer labels
        num_labels = int(mask.max()) + 1
        palette = RandomState(42).randint(0, 255, (num_labels, 3)).astype(np.uint8)
        palette[0] = [0, 0, 0]  # Background is black
        colored_mask = palette[mask.astype(int)]
        colored.append(colored_mask)
    return colored


def colormap_norms_list(norms_list: list) -> list:
    """Convert normal maps to RGB images."""
    colored = []
    for norms in norms_list:
        # norms is (H, W, 3) in range [-1, 1]
        rgb = ((norms + 1) / 2 * 255).astype(np.uint8)
        colored.append(rgb)
    return colored


class SamMeshSegmenter:
    """
    Segments a mesh using the SAMesh model (SAM2-based mesh segmentation).
    Runs directly in the main process for transparency.
    Outputs intermediate render images for debugging/visualization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "sam_checkpoint_path": ("STRING", {"forceInput": True}),
                "sam_model_config_path": ("STRING", {"forceInput": True}),
                "output_directory": ("STRING", {
                    "default": DEFAULT_OUTPUT_DIR,
                    "tooltip": "Directory to save output files."
                }),
                "cache_directory": ("STRING", {
                    "default": DEFAULT_CACHE_DIR,
                    "tooltip": "Directory for caching intermediate results."
                }),
                "keep_texture": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve original mesh texture during segmentation."
                }),
            },
            "optional": {
                "target_labels": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Desired number of segments. -1 for automatic."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible results."
                }),
                "points_per_side": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 64,
                    "step": 8,
                    "tooltip": "SAM point grid density. Lower = faster but coarser. 16=fast, 32=default, 64=fine"
                }),
                "render_resolution": (["1024", "768", "512"], {
                    "default": "1024",
                    "tooltip": "Multiview render resolution. Lower = faster but less detail."
                }),
                "segmentation_mode": (["both", "normals", "sdf"], {
                    "default": "both",
                    "tooltip": "Which render modes to use. 'both' is most robust, single mode is ~2x faster."
                }),
                "camera_radius": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.5,
                    "max": 5.0,
                    "step": 0.5,
                    "tooltip": "Camera distance from mesh center. Higher = mesh smaller in frame but fully visible. Lower = larger but may clip elongated meshes."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("segmented_mesh", "renders_normals", "renders_sdf", "renders_masks", "renders_matte", "stats")
    FUNCTION = "segment_mesh"
    CATEGORY = "meshsegmenter/sammesh"

    def segment_mesh(
        self,
        mesh: trimesh.Trimesh,
        sam_checkpoint_path: str,
        sam_model_config_path: str,
        output_directory: str,
        cache_directory: str,
        keep_texture: bool,
        target_labels: int = -1,
        seed: int = 0,
        points_per_side: int = 32,
        render_resolution: str = "1024",
        segmentation_mode: str = "both",
        camera_radius: float = 3.0
    ):
        from pathlib import Path
        from omegaconf import OmegaConf

        # Import samesh
        print("SamMeshSegmenter: Importing SAMesh...")
        from samesh.models.sam_mesh import segment_mesh as segment_mesh_samesh_func
        from samesh.renderer.renderer import colormap_norms
        print("SamMeshSegmenter: SAMesh imported successfully")

        # Input validation
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Warning: Input 'mesh' is not a Trimesh object (got {type(mesh)})")

        # Setup directories
        os.makedirs(cache_directory, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)

        # Always normalize mesh for samesh processing (cameras assume mesh centered at origin in [-1, 1] bounds)
        # Keep original mesh for output, only normalize the copy for processing
        mesh_extents = mesh.bounding_box.extents
        max_extent = max(mesh_extents)
        print(f"SamMeshSegmenter: Normalizing mesh (extents {mesh_extents[0]:.2f} x {mesh_extents[1]:.2f} x {mesh_extents[2]:.2f}) for processing...")

        mesh_normalized = mesh.copy()
        centroid = mesh_normalized.bounding_box.centroid
        mesh_normalized.vertices -= centroid
        mesh_normalized.vertices /= (max_extent / 2) * 1.001  # Scale to [-1, 1] with small margin

        # Save normalized mesh to temp file for samesh processing
        mesh_path = os.path.join(cache_directory, "input_mesh_temp.glb")
        mesh_normalized.export(mesh_path)

        if not os.path.exists(sam_checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint_path}")
        if not os.path.exists(sam_model_config_path):
            raise FileNotFoundError(f"SAM config not found: {sam_model_config_path}")

        print(f"SamMeshSegmenter: Starting segmentation for: {mesh_path}")

        # Set seed for reproducibility
        capped_seed = seed % (2**32)
        print(f"SamMeshSegmenter: Setting seed to {capped_seed}")
        torch.manual_seed(capped_seed)
        np.random.seed(capped_seed)
        random.seed(capped_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(capped_seed)

        # Parse render resolution
        render_dim = int(render_resolution)

        # Parse segmentation mode
        if segmentation_mode == "both":
            use_modes = ['sdf', 'norms']
        elif segmentation_mode == "normals":
            use_modes = ['norms']
        else:  # sdf
            use_modes = ['sdf']

        # Create configuration
        config = OmegaConf.create({
            "cache": cache_directory,
            "cache_overwrite": False,
            "output": output_directory,
            "sam": {
                "sam": {
                    "checkpoint": sam_checkpoint_path,
                    "model_config": os.path.basename(sam_model_config_path),
                    "auto": True,
                    "ground": False,
                    "engine_config": {
                        "points_per_side": points_per_side,
                        "crop_n_layers": 0,
                        "pred_iou_thresh": 0.5,
                        "stability_score_thresh": 0.7,
                        "stability_score_offset": 1.0,
                    }
                }
            },
            "sam_mesh": {
                "use_modes": use_modes,
                "min_area": 1024,
                "connections_bin_resolution": 100,
                "connections_bin_threshold_percentage": 0.125,
                "smoothing_threshold_percentage_size": 0.025,
                "smoothing_threshold_percentage_area": 0.025,
                "smoothing_iterations": 64,
                "repartition_cost": 1,
                "repartition_lambda": 6,
                "repartition_iterations": 1,
            },
            "renderer": {
                "target_dim": [render_dim, render_dim],
                "camera_generation_method": "icosahedron",
                "renderer_args": {"interpolate_norms": True},
                "sampling_args": {"radius": camera_radius},
                "lighting_args": {}
            }
        })

        # Handle target_labels
        target_labels_arg = None if target_labels < 0 else target_labels
        mesh_file_path = Path(mesh_path)
        filename_stem = mesh_file_path.stem

        print(f"SamMeshSegmenter: Running segmentation...")

        # Run segmentation with renders
        result = segment_mesh_samesh_func(
            filename=mesh_file_path,
            config=config,
            visualize=False,
            extension="glb",
            target_labels=target_labels_arg,
            texture=keep_texture,
            return_renders=True
        )

        # Unpack result
        _, renders = result
        print(f"SamMeshSegmenter: Segmentation complete")

        # Convert renders to ComfyUI image tensors
        print(f"SamMeshSegmenter: Converting {len(renders.get('norms', []))} render views to images...")

        # Normals - convert from [-1,1] to RGB
        if 'norms' in renders and renders['norms']:
            norms_colored = [colormap_norms(n) for n in renders['norms']]
            renders_normals = pil_to_tensor(norms_colored)
        else:
            renders_normals = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # SDF renders (already PIL images)
        if 'sdf' in renders and renders['sdf']:
            renders_sdf = pil_to_tensor(renders['sdf'])
        else:
            renders_sdf = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # Combined masks - colormap them
        if 'cmasks' in renders and renders['cmasks']:
            masks_colored = colormap_masks(renders['cmasks'])
            renders_masks = numpy_to_tensor(masks_colored, normalize=True)
        else:
            renders_masks = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # Matte renders (already PIL images)
        if 'matte' in renders and renders['matte']:
            renders_matte = pil_to_tensor(renders['matte'])
        else:
            renders_matte = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        print(f"SamMeshSegmenter: Render tensors - normals:{renders_normals.shape}, sdf:{renders_sdf.shape}, masks:{renders_masks.shape}, matte:{renders_matte.shape}")

        # Locate samesh output files (in subdirectory)
        output_subdir = os.path.join(output_directory, filename_stem)
        default_mesh_path = os.path.join(output_subdir, f"{filename_stem}_segmented.glb")
        default_json_path = os.path.join(output_subdir, f"{filename_stem}_face2label.json")

        # Load face labels before cleanup
        if not os.path.exists(default_json_path):
            raise FileNotFoundError(f"Face labels not found at: {default_json_path}")
        with open(default_json_path, 'r') as f:
            face2label = json.load(f)

        # Clean up samesh output files (we don't need them)
        if os.path.exists(default_mesh_path):
            os.remove(default_mesh_path)
        os.remove(default_json_path)

        # Clean up empty subdirectory
        try:
            if os.path.exists(output_subdir) and not os.listdir(output_subdir):
                os.rmdir(output_subdir)
        except OSError:
            pass

        # Use the ORIGINAL input mesh (preserves size, vertex count)
        # Just add the segmentation labels as a face attribute
        num_faces = len(mesh.faces)
        seg_labels = np.zeros(num_faces, dtype=np.int32)
        for face_idx, label in face2label.items():
            idx = int(face_idx)
            if idx < num_faces:
                seg_labels[idx] = label

        # Store as face attribute on original mesh
        mesh.face_attributes['seg'] = seg_labels

        # Build stats output
        num_segments = len(np.unique(seg_labels))
        unlabeled_count = np.sum(seg_labels == 0)
        labeled_count = num_faces - unlabeled_count

        stats = f"""Segmentation Stats:
  Total faces: {num_faces:,}
  Labeled: {labeled_count:,} ({100*labeled_count/num_faces:.1f}%)
  Unlabeled: {unlabeled_count:,} ({100*unlabeled_count/num_faces:.1f}%)
  Segments: {num_segments}
  Vertices: {len(mesh.vertices):,}"""

        print(f"SamMeshSegmenter: Done!\n{stats}")

        return (mesh, renders_normals, renders_sdf, renders_masks, renders_matte, stats)
