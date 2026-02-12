# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
PartField Model Downloader Node - Downloads PartField model checkpoint.
"""

import os
import sys
import torch
from huggingface_hub import hf_hub_download

try:
    import folder_paths
    partfield_model_dir = os.path.join(folder_paths.models_dir, "partfield")
except ImportError:
    partfield_model_dir = os.path.join(os.path.expanduser("~"), ".cache", "partfield_models")

os.makedirs(partfield_model_dir, exist_ok=True)


# PartField Model Definitions
PARTFIELD_MODELS = {
    "PartField Objaverse": {
        "checkpoint_filename": "model_objaverse.ckpt",
        "repo_id": "mikaelaangel/partfield-ckpt",
        "description": "Trained on Objaverse dataset (~340k shapes)"
    },
}

PARTFIELD_MODEL_NAMES = list(PARTFIELD_MODELS.keys())


def create_partfield_config():
    """Create default PartField configuration for inference."""
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.seed = 0
    cfg.output_dir = "results"
    cfg.result_name = "test_all"

    cfg.vertex_feature = False
    cfg.n_point_per_face = 2000
    cfg.n_sample_each = 10000
    cfg.preprocess_mesh = False
    cfg.regress_2d_feat = False
    cfg.is_pc = False
    cfg.remesh_demo = False
    cfg.correspondence_demo = False

    cfg.triplane_resolution = 128
    cfg.triplane_channels_low = 128
    cfg.triplane_channels_high = 512

    cfg.use_pvcnn = False
    cfg.use_pvcnnonly = True
    cfg.use_2d_feat = False

    cfg.pvcnn = CN()
    cfg.pvcnn.point_encoder_type = 'pvcnn'
    cfg.pvcnn.use_point_scatter = True
    cfg.pvcnn.z_triplane_channels = 64
    cfg.pvcnn.z_triplane_resolution = 256
    cfg.pvcnn.unet_cfg = CN()
    cfg.pvcnn.unet_cfg.depth = 3
    cfg.pvcnn.unet_cfg.enabled = True
    cfg.pvcnn.unet_cfg.rolled = True
    cfg.pvcnn.unet_cfg.use_3d_aware = True
    cfg.pvcnn.unet_cfg.start_hidden_channels = 32
    cfg.pvcnn.unet_cfg.use_initial_conv = False

    cfg.dataset = CN()
    cfg.dataset.type = "Demo_Dataset"
    cfg.dataset.data_path = ""
    cfg.dataset.train_num_workers = 4
    cfg.dataset.val_num_workers = 4
    cfg.dataset.train_batch_size = 1
    cfg.dataset.val_batch_size = 1
    cfg.dataset.all_files = []

    return cfg


class PartFieldModelDownloader:
    """
    Downloads a PartField model checkpoint and loads it for inference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (PARTFIELD_MODEL_NAMES, {
                    "default": PARTFIELD_MODEL_NAMES[0],
                    "tooltip": "Select the PartField model to download."
                }),
            },
            "optional": {
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device to load the model on."
                }),
            }
        }

    RETURN_TYPES = ("PARTFIELD_MODEL", "STRING",)
    RETURN_NAMES = ("model", "checkpoint_path",)
    FUNCTION = "download_and_load_model"
    CATEGORY = "meshsegmenter/partfield"

    def download_and_load_model(self, model_name: str, device: str = "cuda"):
        """Download and load the specified PartField model."""
        if model_name not in PARTFIELD_MODELS:
            raise ValueError(f"Selected model '{model_name}' is not defined.")

        model_info = PARTFIELD_MODELS[model_name]
        checkpoint_filename = model_info["checkpoint_filename"]
        repo_id = model_info["repo_id"]

        checkpoint_path = os.path.join(partfield_model_dir, checkpoint_filename)

        # Download Checkpoint if missing
        if not os.path.exists(checkpoint_path):
            print(f"PartFieldModelDownloader ({model_name}): Checkpoint not found. Downloading...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=checkpoint_filename,
                    local_dir=partfield_model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print(f"PartFieldModelDownloader ({model_name}): Checkpoint downloaded to {checkpoint_path}")
            except Exception as e:
                print(f"\033[31mError downloading checkpoint for {model_name}: {e}\033[0m")
                raise
        else:
            print(f"PartFieldModelDownloader ({model_name}): Checkpoint found: {checkpoint_path}")

        # Verify file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Failed to locate checkpoint: {checkpoint_path}")

        # Load the model
        print(f"PartFieldModelDownloader ({model_name}): Loading model...")

        # Load checkpoint to get hyperparameters
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract config from checkpoint if available, otherwise use defaults
        if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
            cfg = checkpoint['hyper_parameters']['cfg']
            print(f"PartFieldModelDownloader ({model_name}): Using config from checkpoint")
        else:
            cfg = create_partfield_config()
            print(f"PartFieldModelDownloader ({model_name}): Using default config")

        # Import and instantiate model
        from ...partfield.model_trainer_pvcnn_only_demo import Model

        # Create model instance
        model = Model(cfg)

        # Load state dict from checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'model.' prefix if present (from Lightning)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)
        model.eval()

        print(f"PartFieldModelDownloader ({model_name}): Model loaded successfully on {device}")

        # Return model wrapped with config and device info
        model_wrapper = {
            'model': model,
            'config': cfg,
            'device': device,
            'checkpoint_path': checkpoint_path
        }

        return (model_wrapper, checkpoint_path,)
