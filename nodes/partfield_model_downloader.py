# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-MeshSegmenter Contributors

"""
PartField Model Downloader Node - Downloads PartField model checkpoint.
"""

import logging
import os

import torch
from huggingface_hub import hf_hub_download
from folder_paths import base_path as comfy_base_path

log = logging.getLogger("meshsegmenter")

partfield_model_dir = os.path.join(comfy_base_path, "models", "partfield")
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
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
            },
        }

    RETURN_TYPES = ("PARTFIELD_MODEL", "STRING",)
    RETURN_NAMES = ("model", "checkpoint_path",)
    FUNCTION = "download_and_load_model"
    CATEGORY = "meshsegmenter/partfield"

    def _resolve_dtype(self, precision, device):
        import comfy.model_management
        if precision == "auto":
            if comfy.model_management.should_use_bf16(device):
                return torch.bfloat16
            elif comfy.model_management.should_use_fp16(device):
                return torch.float16
            else:
                return torch.float32
        elif precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        else:
            return torch.float32

    def download_and_load_model(self, model_name: str, precision: str = "auto"):
        """Download and load the specified PartField model."""
        import comfy.model_management

        if model_name not in PARTFIELD_MODELS:
            raise ValueError(f"Selected model '{model_name}' is not defined.")

        model_info = PARTFIELD_MODELS[model_name]
        checkpoint_filename = model_info["checkpoint_filename"]
        repo_id = model_info["repo_id"]

        checkpoint_path = os.path.join(partfield_model_dir, checkpoint_filename)

        # Download Checkpoint if missing
        if not os.path.exists(checkpoint_path):
            log.info("PartFieldModelDownloader (%s): Checkpoint not found. Downloading...", model_name)
            hf_hub_download(
                repo_id=repo_id,
                filename=checkpoint_filename,
                local_dir=partfield_model_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            log.info("PartFieldModelDownloader (%s): Checkpoint downloaded to %s", model_name, checkpoint_path)
        else:
            log.info("PartFieldModelDownloader (%s): Checkpoint found: %s", model_name, checkpoint_path)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Failed to locate checkpoint: {checkpoint_path}")

        # Resolve device and dtype
        load_device = comfy.model_management.get_torch_device()
        dtype = self._resolve_dtype(precision, load_device)
        device_str = str(load_device)

        log.info("PartFieldModelDownloader (%s): Loading model...", model_name)

        # Load checkpoint to get hyperparameters
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract config from checkpoint if available, otherwise use defaults
        if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
            cfg = checkpoint['hyper_parameters']['cfg']
            log.info("PartFieldModelDownloader (%s): Using config from checkpoint", model_name)
        else:
            cfg = create_partfield_config()
            log.info("PartFieldModelDownloader (%s): Using default config", model_name)

        # Import and instantiate model
        from .partfield_lib.model_trainer_pvcnn_only_demo import Model

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
        model = model.to(device_str)
        model.eval()

        log.info("PartFieldModelDownloader (%s): Model loaded on %s (%s)", model_name, device_str, dtype)

        # Return model wrapped with config and device info
        model_wrapper = {
            'model': model,
            'config': cfg,
            'device': device_str,
            'checkpoint_path': checkpoint_path
        }

        return (model_wrapper, checkpoint_path,)


NODE_CLASS_MAPPINGS = {
    "MeshSegPartFieldModelDownloader": PartFieldModelDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshSegPartFieldModelDownloader": "PartField Model Downloader (MeshSeg)",
}
